"""SRE debugging tools exposed to the model via function calling.

Each tool has:
  - a JSON-schema spec (sent to the LLM in the `tools` field), and
  - a Python implementation in TOOL_IMPLS keyed by tool name.

Design notes:
  - `run_shell` is the workhorse (kubectl, curl, dig, ps, journalctl, ...).
    It is read-only *by convention only* — the agent layer asks the user to
    approve each command unless AUTO_APPROVE is set. A small denylist blocks
    the most obviously destructive patterns regardless.
  - All output is truncated so a noisy command can't blow up the context.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import urllib.request
import urllib.error
from time import perf_counter

MAX_OUTPUT_CHARS = 16000
DEFAULT_TIMEOUT = 60

# ── Last-ditch denylist (backstop, NOT the real guard) ──────────────────────
# Catastrophic, irreversible commands that should never run through this tool,
# even when a human approves them by reflex. This is a seatbelt, not a wall:
# you cannot reliably block destructive shell with regex (flag reordering,
# `bash -c`, `$(…)`, env indirection all slip past). The approval prompt and
# the read-only allowlist below are what actually keep auto-mode safe.
_DESTRUCTIVE = re.compile(
    r"\brm\s+(-[a-z]*r|--recursive)"        # recursive delete, any flag order/case
    r"|--no-preserve-root"
    r"|\bfind\b[^\n]*\s-delete\b"
    r"|\bmkfs\b|\bdd\s+if=|\bdd\b[^\n]*\bof=/dev/"
    r"|:\(\)\s*\{"                          # fork bomb
    r"|\b(shutdown|reboot|halt|poweroff)\b"
    r"|>\s*/dev/(sd|nvme|xvd)"              # clobber a block device
    r"|\bkubectl\s+delete\b"
    r"|\bterraform\s+destroy\b"
    r"|\bhelm\s+(uninstall|delete)\b",
    re.IGNORECASE,
)

# ── Read-only allowlist: the real guard for AUTO_APPROVE ─────────────────────
# Under AUTO_APPROVE, only commands that pass is_auto_safe() run without a
# prompt. Everything else (any mutation, anything we can't vouch for) still
# prompts — so a non-interactive run declines it rather than executing blindly.

# Shell constructs that could chain/hide/redirect a second command. If any are
# present, the command is NOT auto-safe (it must be a single, simple command).
_SHELL_METACHARS = re.compile(r"[;&|`<>\n()]|\$[({]")

# kubectl: allow only if it contains a read verb and NO mutating verb.
_KUBECTL_READ_VERBS = {
    "get", "describe", "logs", "top", "version", "explain", "events",
    "api-resources", "api-versions", "cluster-info", "config", "auth",
}
_KUBECTL_WRITE_VERBS = {
    "delete", "drain", "cordon", "uncordon", "taint", "scale", "patch",
    "replace", "apply", "edit", "set", "rollout", "annotate", "label",
    "create", "run", "expose", "autoscale", "exec", "cp", "attach",
    "port-forward", "proxy", "debug",
}
_SYSTEMCTL_WRITE_VERBS = {
    "start", "stop", "restart", "reload", "enable", "disable", "mask",
    "unmask", "kill", "isolate", "set-property", "daemon-reload", "edit",
}
# Plain binaries that only read/observe.
_SAFE_BINARIES = {
    "curl", "wget", "dig", "nslookup", "host", "getent",
    "ps", "top", "free", "df", "du", "uptime", "vmstat", "iostat", "mpstat",
    "journalctl", "dmesg", "uname", "hostname", "whoami", "id", "date",
    "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "wc", "zcat",
    "ls", "stat", "find", "sort", "uniq", "cut", "tr", "column",
    "ss", "netstat", "ip", "ping", "nproc", "lscpu", "lsblk",
    "echo", "printenv", "env", "true",
}
# Flags that turn an otherwise-safe binary into a writer/executor.
_DANGEROUS_FLAGS = {
    "sed": {"-i", "--in-place"},
    "find": {"-delete", "-exec", "-execdir", "-fprint", "-fprintf", "-fls"},
    "curl": {"-o", "-O", "--output", "--upload-file", "-T",
             "-X", "--request", "-d", "--data", "--data-binary", "--data-raw"},
    "wget": {"-O", "--output-document", "--post-data", "--post-file"},
    "env": {"-i"},  # `env -i` resets environment; also used to launch programs
}


def is_auto_safe(cmd: str) -> bool:
    """Whether `cmd` is a single read-only command safe to auto-run under
    AUTO_APPROVE. Conservative: anything not clearly read-only returns False,
    and the caller asks for confirmation instead of running it."""
    if not cmd or _DESTRUCTIVE.search(cmd) or _SHELL_METACHARS.search(cmd):
        return False
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return False
    if not tokens:
        return False
    binary = os.path.basename(tokens[0])
    rest = set(tokens[1:])

    if binary == "kubectl":
        if rest & _KUBECTL_WRITE_VERBS:
            return False
        return bool(rest & _KUBECTL_READ_VERBS)
    if binary == "systemctl":
        return not (rest & _SYSTEMCTL_WRITE_VERBS)
    if rest & _DANGEROUS_FLAGS.get(binary, set()):
        return False
    return binary in _SAFE_BINARIES


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text)} chars total]"
    return text


def run_shell(cmd: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a shell command and return combined stdout/stderr + exit code."""
    if _DESTRUCTIVE.search(cmd):
        return "REFUSED: command matches a destructive pattern and was not run."
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=min(int(timeout), 600),
        )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s: {cmd}"
    out = proc.stdout or ""
    err = proc.stderr or ""
    body = out
    if err:
        body += ("\n[stderr]\n" + err) if body else ("[stderr]\n" + err)
    return _truncate(f"exit_code={proc.returncode}\n{body}".strip())


def read_file(path: str, start_line: int = 1, end_line: int = 400) -> str:
    """Read a slice of a text file (1-indexed, inclusive)."""
    try:
        with open(os.path.expanduser(path), "r", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"ERROR: {e}"
    start = max(1, int(start_line))
    end = min(len(lines), int(end_line))
    chunk = "".join(f"{i:>6}|{lines[i - 1]}" for i in range(start, end + 1))
    return _truncate(chunk or "(empty range)")


def http_check(url: str, timeout: int = 10) -> str:
    """GET a URL and report status code + latency + first bytes of the body."""
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "sre-agent"})
    t0 = perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=min(int(timeout), 30)) as resp:
            body = resp.read(2000).decode("utf-8", errors="replace")
            ms = (perf_counter() - t0) * 1000
            return f"status={resp.status} latency={ms:.0f}ms\n{body}"
    except urllib.error.HTTPError as e:
        ms = (perf_counter() - t0) * 1000
        return f"status={e.code} latency={ms:.0f}ms (HTTPError)"
    except Exception as e:  # noqa: BLE001 — surface any connection error to the model
        ms = (perf_counter() - t0) * 1000
        return f"ERROR after {ms:.0f}ms: {type(e).__name__}: {e}"


# ───  Specs sent to the model (OpenAI tool schema) ───

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command for diagnostics (kubectl, curl, dig, ps, top, "
                "journalctl, df, free, systemctl status, etc.). Prefer read-only "
                "commands. Returns exit code and combined stdout/stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "The shell command to run."},
                    "timeout": {"type": "integer", "description": "Seconds before timeout (max 600)."},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a slice of a text file (logs, configs, manifests).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or ~-relative file path."},
                    "start_line": {"type": "integer", "description": "1-indexed start line (default 1)."},
                    "end_line": {"type": "integer", "description": "Inclusive end line (default 400)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_check",
            "description": "GET a URL and report HTTP status, latency, and a snippet of the body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL including scheme."},
                    "timeout": {"type": "integer", "description": "Seconds before timeout (max 30)."},
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_IMPLS = {
    "run_shell": run_shell,
    "read_file": read_file,
    "http_check": http_check,
}

# Tools that mutate the system / run arbitrary commands need user approval.
NEEDS_APPROVAL = {"run_shell"}
