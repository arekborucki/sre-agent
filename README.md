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

## Incident memory (Qdrant)

The agent remembers **resolved** incidents so it can recall a prior root cause
before re-investigating from scratch — same symptom next time, much faster
answer. Each resolved incident (symptom, environment, signals, root cause, fix)
is stored as a point in a Qdrant collection (`sre-incidents`). At the start of a
new investigation the agent searches that memory for similar past incidents and
starts from what already worked.

Search is **hybrid**, with two named vectors per incident:

- **`e5`** — dense, `intfloat/multilingual-e5-small` (384-dim, cosine). Matches by
  *meaning*, across Polish and English — "pod się restartuje" finds a past
  "CrashLoopBackOff" incident.
- **`bm25`** — sparse, `qdrant/bm25` (IDF). Matches exact tokens — error codes,
  namespace names, signals (`OOMKilled`, `exit_code=137`).

Results are fused with Reciprocal Rank Fusion (RRF), so you get both the
semantically-similar and the keyword-exact hits.

### Why Qdrant

Recalling a similar past incident is a **search** problem, not a storage problem,
and the quality of that search is what makes the memory useful. Qdrant fits for
concrete reasons:

- **Hybrid search is native.** Dense (meaning) + sparse BM25 (keywords) + RRF
  fusion in a single query. Plain files or a relational table would force us to
  bolt this together by hand.
- **Auto-embedding (Cloud Inference) keeps the agent thin.** With
  `cloud_inference=True` you send the raw symptom *text*; the cluster computes
  the embedding server-side. The agent needs no embedding model, no GPU, no
  extra dependency — it just sends strings.
- **Managed, so no ops burden.** A managed Qdrant cluster means no JVM heap to
  tune, no shards/ILM to babysit, no version upgrades to run — unlike standing
  up Elasticsearch for the same job. The free tier is plenty for an incident
  base that starts empty and grows slowly.
- **Multilingual by design.** The `multilingual-e5-small` model handles the
  mixed PL/EN way incidents actually get described.

The storage layer is intentionally isolated behind a small interface
(`save_incident` / `search_incidents`), so if the base ever outgrows Qdrant the
backend can be swapped without touching the agent or its tools.

### Setup

```bash
pip install -r requirements.txt        # adds qdrant-client
cp .env.example .env                    # fill QDRANT_URL + QDRANT_API_KEY
python setup_qdrant.py                  # one-time: creates the sre-incidents collection
```

- `QDRANT_URL` — your managed cluster endpoint on **port 6333**.
- `QDRANT_API_KEY` — a **read-write Database API key** (created in the cluster's
  API Keys panel), not a cloud-management key.
- Cloud Inference must be enabled on the cluster (Inference tab) — it lists the
  E5 and BM25 models when it is.

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
| `QDRANT_URL` | — | Managed Qdrant cluster endpoint (port 6333). Required for incident memory. |
| `QDRANT_API_KEY` | — | Read-write Database API key for the cluster. |
| `QDRANT_COLLECTION` | `sre-incidents` | Collection holding resolved incidents. |
