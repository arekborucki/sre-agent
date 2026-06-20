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
import subprocess
import urllib.request
import urllib.error
from time import perf_counter

MAX_OUTPUT_CHARS = 16000
DEFAULT_TIMEOUT = 60

# Block the most catastrophic patterns even when auto-approve is on.
_DESTRUCTIVE = re.compile(
    r"\brm\s+-rf\s+/|\bmkfs\b|\bdd\s+if=|\b:\(\)\s*\{|\bshutdown\b|\breboot\b|\bhalt\b"
    r"|>\s*/dev/sd|\bkubectl\s+delete\b|\bterraform\s+(destroy|apply)\b"
)


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
