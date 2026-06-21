"""A minimal SRE debugging agent.

A small tool-calling loop, ~150 lines of Python:
  send prompt + tools -> model asks for a tool -> we run it -> feed result back
  -> loop until the model answers in plain text.

Provider: Hugging Face router (OpenAI-compatible), so we just point the
official `openai` SDK at https://router.huggingface.co/v1.

Run:
    pip install -r requirements.txt
    cp .env.example .env   # fill in HF_TOKEN
    python agent.py "why is my pod CrashLoopBackOff in namespace prod?"
    python agent.py        # interactive REPL
"""

from __future__ import annotations

import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from skills import skills_catalog
from tools import TOOLS_SPEC, TOOL_IMPLS, NEEDS_APPROVAL, is_auto_safe

load_dotenv()

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
MAX_ITERATIONS = 20
# Each API call resends the whole history, so old tool outputs (kubectl/logs/
# curl dumps) pile up and cost grows quadratically. Keep the N most recent tool
# results in full; shrink older ones to a stub (see shrink_old_outputs).
KEEP_RECENT_TOOL_OUTPUTS = 6
# Cap a single model reply so one verbose answer can't blow up the context.
MAX_RESPONSE_TOKENS = 2000
# The HF router can return transient timeouts or 5xx (e.g. a cold model warming
# up). Retry a few times with exponential backoff so one blip doesn't kill the
# turn, and bound each request so it can't hang forever.
API_TIMEOUT_SECONDS = 60
API_MAX_RETRIES = 4

SYSTEM_PROMPT = """You are an SRE debugging assistant operating in a terminal.

Your job: diagnose infrastructure and application incidents by gathering
evidence with the tools, forming a hypothesis, and confirming it before
concluding. You are a careful, read-first operator.

Guidelines:
- FIRST, before investigating, call search_incidents with the symptom. If a
  past incident matches, start from its root cause and fix — verify it still
  applies rather than re-deriving from zero. This is how you get faster over
  time. If nothing matches, investigate from scratch.
- Investigate with read-only commands first (get/describe/logs/top, curl,
  dig, systemctl status, journalctl, df, free). Never mutate state unless the
  user explicitly asks and approves.
- Work in small steps: one focused command at a time, then reason about the
  output before the next.
- When you reach a conclusion, give: (1) root cause, (2) the evidence that
  proves it, (3) a concrete fix or next action. Be concise.
- Once the root cause is confirmed and a fix is known, call save_incident so
  the next investigation can reuse it. Write the title and fields in ENGLISH
  (the title becomes the archive filename). Capture distinctive signals (error
  codes, OOMKilled, exit codes) so similar incidents are easy to recall.
- If a tool fails with auth/permission/missing-binary errors, stop and tell
  the user — that's an environment problem, not something to brute-force.
"""


def approve(name: str, args: dict) -> bool:
    """Decide whether to run a tool, prompting the user when needed.

    AUTO_APPROVE only auto-runs vetted read-only commands (see is_auto_safe);
    anything that could mutate state still prompts — so a non-interactive run
    declines it rather than executing blindly. The prompt, not the denylist,
    is the real safety boundary.
    """
    if name not in NEEDS_APPROVAL:
        return True
    auto = os.getenv("AUTO_APPROVE", "false").lower() == "true"
    cmd = args.get("cmd", "")
    if auto and name == "run_shell" and is_auto_safe(cmd):
        return True
    label = cmd or json.dumps(args)
    if auto:
        print("\n  \033[2m(AUTO_APPROVE on, but this isn't a vetted read-only command)\033[0m")
    print(f"  \033[33m? {name}\033[0m {label}")
    try:
        return input("  run it? [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def run_tool(name: str, args: dict) -> str:
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return f"ERROR: unknown tool {name}"
    if not approve(name, args):
        return "User declined to run this command."
    try:
        return impl(**args)
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
    except Exception as e:  # noqa: BLE001 — return errors to the model, don't crash
        return f"ERROR running {name}: {type(e).__name__}: {e}"


def shrink_old_outputs(messages: list) -> None:
    """Replace the body of OLD tool outputs with a short stub, in place.

    We never drop messages: the chat protocol requires every assistant message
    that has tool_calls to be followed by `tool` messages with matching
    tool_call_ids, so removing one would orphan the pair and the API would 400.
    Shrinking only the *content* keeps that structure intact while removing the
    bulk. The model has already reasoned over those old outputs (its conclusions
    live in the assistant messages, which we keep), so the raw text is dead
    weight after KEEP_RECENT_TOOL_OUTPUTS newer results exist.
    """
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idxs[:-KEEP_RECENT_TOOL_OUTPUTS] if KEEP_RECENT_TOOL_OUTPUTS else tool_idxs:
        body = messages[i].get("content") or ""
        if not body.startswith("[older tool output elided"):
            messages[i]["content"] = f"[older tool output elided, {len(body)} chars]"


# Transient errors worth retrying; anything else (bad request, auth) is fatal.
_TRANSIENT_API_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


def chat_with_retry(client: OpenAI, model: str, messages: list):
    """Call the chat API, retrying transient errors with exponential backoff.
    Raises the last error if every attempt fails."""
    delay = 2
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS_SPEC,
                temperature=0.0,
                max_tokens=MAX_RESPONSE_TOKENS,
            )
        except _TRANSIENT_API_ERRORS as e:
            if attempt == API_MAX_RETRIES:
                raise
            print(
                f"  \033[2m(transient API error: {type(e).__name__}; "
                f"retry {attempt}/{API_MAX_RETRIES - 1} in {delay}s)\033[0m",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2


def run_turn(client: OpenAI, model: str, messages: list) -> str:
    """Drive one user turn to completion (the tool loop)."""
    for _ in range(MAX_ITERATIONS):
        shrink_old_outputs(messages)  # keep context (and cost) bounded
        try:
            resp = chat_with_retry(client, model, messages)
        except Exception as e:  # noqa: BLE001 — degrade gracefully, no traceback at the user
            return f"Stopped: the model API is unavailable after retries ({type(e).__name__}: {e}). Try again shortly."
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or "(no output)"

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"  \033[36m> {name}\033[0m {args.get('cmd', '') or json.dumps(args)}")
            result = run_tool(name, args)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    return "Stopped: hit the tool-iteration limit without a final answer."


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set. Copy .env.example to .env and fill it in.")
    model = os.getenv("MODEL_ID", "moonshotai/Kimi-K2-Instruct")
    # We handle retries ourselves (with backoff), so disable the SDK's own.
    client = OpenAI(
        base_url=HF_ROUTER_BASE_URL,
        api_key=token,
        timeout=API_TIMEOUT_SECONDS,
        max_retries=0,
    )

    # Append the skills catalog (names + descriptions only) to the system prompt.
    # The model loads a skill's full body on demand via the load_skill tool.
    messages = [{"role": "system", "content": SYSTEM_PROMPT + skills_catalog()}]
    one_shot = " ".join(sys.argv[1:]).strip()

    if one_shot:
        messages.append({"role": "user", "content": one_shot})
        print(run_turn(client, model, messages))
        return

    print(f"SRE agent ready (model: {model}). Type a question, Ctrl-C to quit.\n")
    while True:
        try:
            user = input("\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        messages.append({"role": "user", "content": user})
        print(run_turn(client, model, messages) + "\n")


if __name__ == "__main__":
    main()
