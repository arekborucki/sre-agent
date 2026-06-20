# sre-agent

A minimal SRE debugging agent in Python. It uses a Hugging Face model (via the
OpenAI-compatible HF router) and a small set of diagnostic tools. The model
investigates an incident by running read-only commands, reasoning over the
output, and proposing a root cause + fix.

It's the same loop moon-bot uses, stripped to the essentials: **send prompt +
tools → model asks for a tool → run it → feed the result back → repeat until
the model answers in plain text.**

## Setup

```bash
cd sre-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your HF_TOKEN in .env
```

Get a token at https://huggingface.co/settings/tokens.

## Usage

One-shot:

```bash
python agent.py "pod api-7xx is CrashLoopBackOff in namespace prod, why?"
```

Interactive REPL:

```bash
python agent.py
```

## Tools

| Tool | What it does |
|---|---|
| `run_shell` | Run a diagnostic command (kubectl, curl, dig, journalctl, df, ...). |
| `read_file` | Read a slice of a log / config / manifest. |
| `http_check` | GET a URL → status code, latency, body snippet. |

## Safety

The real safety boundary is the **approval prompt**, not a denylist — you
can't reliably block destructive shell commands with pattern matching
(`kubectl scale --replicas=0`, `sed -i`, `find -delete`, `bash -c '…'`,
`$(…)`, env indirection — all slip past regex). So the model is built around
*what auto-runs*, not *what's forbidden*:

- **Interactive (default):** `run_shell` asks before **every** command.
- **`AUTO_APPROVE=true`:** auto-runs **only vetted read-only commands** — a
  *single* command (no pipes, redirection, `;`/`&&`, `$(…)`) whose binary is on
  a read-only allowlist (`kubectl get/describe/logs/top`, `curl`, `dig`,
  `journalctl`, `df`, `ps`, …; `kubectl`/`systemctl` are checked for mutating
  verbs). **Anything else still prompts** — so in a non-interactive run it is
  *declined*, not executed. Auto-approve does **not** mean "run anything."
- A **denylist** is a last-ditch backstop for catastrophic, irreversible
  commands (`rm -r`, `mkfs`, `dd`, `kubectl delete`, `terraform destroy`,
  `helm uninstall`, reboot/shutdown, …). It runs even on manually-approved
  commands — but it's a seatbelt, **not** a guarantee: not exhaustive, and not
  what keeps auto-mode safe (the allowlist is).
- All tool output is truncated so a noisy command can't blow up the context.

## Extending

Add a tool in `tools.py`: write the function, add its JSON spec to
`TOOLS_SPEC`, and register it in `TOOL_IMPLS`. If it mutates state, add its
name to `NEEDS_APPROVAL`. That's it — the agent loop picks it up automatically.

## Config

| Env var | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token (required). |
| `MODEL_ID` | `moonshotai/Kimi-K2-Instruct` | Any tool-capable model on the HF router. Pin a provider with `model:provider`. |
| `AUTO_APPROVE` | `false` | Auto-run only vetted **read-only** `run_shell` commands; anything that could mutate still prompts. |
