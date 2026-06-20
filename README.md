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

- `run_shell` asks for confirmation before **every** command. Set
  `AUTO_APPROVE=true` in `.env` to skip the prompt (only when you trust it).
- A denylist refuses obviously destructive commands (`rm -rf /`, `mkfs`,
  `kubectl delete`, `terraform destroy/apply`, reboot/shutdown, ...) even with
  auto-approve on.
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
| `AUTO_APPROVE` | `false` | Skip per-command confirmation for `run_shell`. |
