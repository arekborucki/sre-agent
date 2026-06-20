# sre-agent

A minimal SRE debugging agent in Python. It uses a Hugging Face model (via the
OpenAI-compatible HF router) and a small set of diagnostic tools. The model
investigates an incident by running read-only commands, reasoning over the
output, and proposing a root cause + fix.

The loop is deliberately minimal: **send prompt + tools → model asks for a tool
→ run it → feed the result back → repeat until the model answers in plain
text.**

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
| `http_check` | GET a URL, returning status code, latency, and a body snippet. |
| `search_incidents` | Recall similar past incidents from memory (see Incident memory). |
| `save_incident` | Store a resolved incident for future recall. |
| `load_skill` | Load a best-practice playbook on demand (see Skills). |

## Safety

The real safety boundary is the **approval prompt**, not a denylist. You
can't reliably block destructive shell commands with pattern matching.
Things like `kubectl scale --replicas=0`, `sed -i`, `find -delete`,
`bash -c '…'`, `$(…)`, and env indirection all slip past regex. So the model
is built around *what auto-runs*, not *what's forbidden*:

- **Interactive (default):** `run_shell` asks before **every** command.
- **`AUTO_APPROVE=true`:** auto-runs **only vetted read-only commands**. That
  means a *single* command (no pipes, redirection, `;`/`&&`, `$(…)`) whose
  binary is on a read-only allowlist (`kubectl get/describe/logs/top`, `curl`,
  `dig`, `journalctl`, `df`, `ps`, …; `kubectl`/`systemctl` are checked for
  mutating verbs). **Anything else still prompts**, so in a non-interactive run
  it is *declined*, not executed. Auto-approve does **not** mean "run anything."
- A **denylist** is a last-ditch backstop for catastrophic, irreversible
  commands (`rm -r`, `mkfs`, `dd`, `kubectl delete`, `terraform destroy`,
  `helm uninstall`, reboot/shutdown, …). It runs even on manually-approved
  commands, but it's a seatbelt, **not** a guarantee: not exhaustive, and not
  what keeps auto-mode safe (the allowlist is).
- All tool output is truncated so a noisy command can't blow up the context.

## Incident memory (Qdrant)

The agent remembers **resolved** incidents so it can recall a prior root cause
before re-investigating from scratch. The same symptom next time gets a much
faster answer. Each resolved incident (symptom, environment, signals, root
cause, fix) is stored as a point in a Qdrant collection (`sre-incidents`). At
the start of a new investigation the agent searches that memory for similar past
incidents and starts from what already worked.

### How it resolves an incident

The system prompt makes the agent follow this loop on every question:

1. **Recall first.** Call `search_incidents` with the symptom. If a past incident
   matches, start from its `root_cause` and `fix` and verify they still apply,
   rather than re-deriving from zero.
2. **Investigate.** If nothing matches, work the problem with read-only commands
   one step at a time, reasoning over each output before the next.
3. **Conclude.** Report the root cause, the evidence that proves it, and a
   concrete fix or next action.
4. **Remember.** Once the root cause is confirmed and a fix is known, call
   `save_incident` with distinctive `signals` (error codes, `OOMKilled`, exit
   codes) so the next occurrence is easy to recall.

The payoff: the more incidents the agent resolves, the faster it resolves the
next similar one, because step 1 short-circuits the investigation.

Search is **hybrid**, with two named vectors per incident:

- **`e5`** is dense, `intfloat/multilingual-e5-small` (384-dim, cosine). It
  matches by *meaning*, across Polish and English, so "pod się restartuje" finds
  a past "CrashLoopBackOff" incident.
- **`bm25`** is sparse, `qdrant/bm25` (IDF). It matches exact tokens such as
  error codes, namespace names, and signals (`OOMKilled`, `exit_code=137`).

Results are fused with Reciprocal Rank Fusion (RRF), so you get both the
semantically-similar and the keyword-exact hits.

### Schema

Collection `sre-incidents` holds one point per resolved incident. Each point has
two named vectors and a JSON payload.

| Vector | Type | Model | Config |
|---|---|---|---|
| `e5` | dense | `intfloat/multilingual-e5-small` | size 384, distance Cosine |
| `bm25` | sparse | `qdrant/bm25` | modifier IDF |

Both vectors are produced from the same indexed text, a join of `title`,
`symptom`, and `signals`, embedded server-side by Cloud Inference.

Payload fields:

```jsonc
{
  "title":        "string",   // short symptom headline, e.g. "api-7xx CrashLoopBackOff in prod"
  "symptom":      "string",   // full description of the observed symptoms (the text we embed)
  "root_cause":   "string",   // confirmed root cause
  "fix":          "string",   // remediation that resolved it
  "environment":  {           // free-form context object
    "cluster":    "string",
    "namespace":  "string",
    "service":    "string"
  },
  "signals":      ["string"], // distinctive tags, e.g. ["OOMKilled", "exit_code=137"]
  "commands_run": ["string"], // key diagnostic commands that found the cause
  "timestamp":    "string"    // ISO 8601 UTC, set automatically on save
}
```

The point `id` is a generated UUID (hex). Only `title`, `symptom`, `root_cause`,
and `fix` are required; the rest are optional.

### Why Qdrant

Recalling a similar past incident is a **search** problem, not a storage problem,
and the quality of that search is what makes the memory useful. Qdrant fits for
concrete reasons:

- **Hybrid search is native.** Dense (meaning) + sparse BM25 (keywords) + RRF
  fusion in a single query. Plain files or a relational table would force us to
  bolt this together by hand.
- **Auto-embedding (Cloud Inference) keeps the agent thin.** With
  `cloud_inference=True` you send the raw symptom *text*; the cluster computes
  the embedding server-side. The agent needs no embedding model, no GPU, and no
  extra dependency. It just sends strings.
- **Managed, so no ops burden.** A managed Qdrant cluster means no JVM heap to
  tune, no shards/ILM to babysit, and no version upgrades to run, unlike
  standing up Elasticsearch for the same job. The free tier is plenty for an
  incident base that starts empty and grows slowly.
- **Multilingual by design.** The `multilingual-e5-small` model handles the
  mixed PL/EN way incidents actually get described.

The storage layer is intentionally isolated behind a small interface
(`save_incident` / `search_incidents`), so if the base ever outgrows Qdrant the
backend can be swapped without touching the agent or its tools.

### Optional: archive to a HF Storage Bucket

When `HF_INCIDENTS_BUCKET` is set, each resolved incident is also written to a
private Hugging Face Storage Bucket as `incidents/<id>.json`, a durable copy
alongside Qdrant.

**What a HF Storage Bucket is.** A bucket is a repo type on the Hugging Face Hub
that provides S3-like object storage, powered by the Xet backend. Unlike a
git-based dataset repo, a bucket is non-versioned and mutable, built for simple
fast storage of files such as logs, artifacts, or any large collection that does
not need version control. It has per-TB pricing, a built-in CDN, Xet
deduplication, and no git overhead, so writes are immediate with no commit queue.

**Why store in both Qdrant and a bucket.** The two systems solve different
problems, and pairing them gives each job the right tool:

- **Qdrant is the search index.** Recalling a similar past incident is a search
  problem, and Qdrant answers it with hybrid dense + sparse retrieval. That is
  what powers `search_incidents`. A bucket cannot do semantic search.
- **The bucket is the durable archive.** It is a cheap, plain copy of every
  incident as readable JSON, independent of Qdrant. If the Qdrant collection is
  lost or rebuilt, the incidents still exist as objects you can re-ingest, and a
  human can read `incidents/<id>.json` directly without any vector tooling.
- **Portability and sharing.** The archive lives under your own HF namespace,
  reachable from any cloud or teammate, decoupled from the search backend.

The bucket write is best-effort: if it fails, the save still succeeds because
Qdrant already holds the incident, and a warning is logged. Buckets are
non-versioned and mutable, so re-saving the same id overwrites the previous
object. Leave `HF_INCIDENTS_BUCKET` empty to run on Qdrant alone.

### Setup

```bash
pip install -r requirements.txt        # adds qdrant-client
cp .env.example .env                    # fill QDRANT_URL + QDRANT_API_KEY
python setup_qdrant.py                  # one-time: creates the sre-incidents collection
```

- `QDRANT_URL` is your managed cluster endpoint on **port 6333**.
- `QDRANT_API_KEY` is a **read-write Database API key** (created in the cluster's
  API Keys panel), not a cloud-management key.
- Cloud Inference must be enabled on the cluster (Inference tab). It lists the
  E5 and BM25 models when it is.

## Skills

Skills are curated best-practice playbooks the model can pull in on demand. Each
skill is a markdown file in `skills/` with simple frontmatter:

```markdown
---
name: kubectl
description: Diagnosing Kubernetes problems with read-only commands.
---
# ...full playbook body...
```

The mechanism is progressive disclosure, the same idea as Claude Code skills.
Only the **catalog** (each skill's name and description) is appended to the
system prompt, which is cheap. When the model judges a skill relevant, it calls
the `load_skill` tool to read that skill's full body, so the body costs tokens
only when actually used. This keeps the prompt small and scales to many skills.

To add a skill, drop a new `.md` file in `skills/`. No code change needed. The
shipped `kubectl` skill covers CrashLoopBackOff, OOMKilled, Pending, ImagePull,
node NotReady, and DNS/Service debugging.

## Extending

Add a tool in `tools.py`: write the function, add its JSON spec to
`TOOLS_SPEC`, and register it in `TOOL_IMPLS`. If it mutates state, add its
name to `NEEDS_APPROVAL`. That's it. The agent loop picks it up automatically.

## Config

| Env var | Default | Description |
|---|---|---|
| `HF_TOKEN` | none | Hugging Face token (required). |
| `MODEL_ID` | `moonshotai/Kimi-K2-Instruct` | Any tool-capable model on the HF router. Pin a provider with `model:provider`. |
| `AUTO_APPROVE` | `false` | Auto-run only vetted **read-only** `run_shell` commands; anything that could mutate still prompts. |
| `QDRANT_URL` | none | Managed Qdrant cluster endpoint (port 6333). Required for incident memory. |
| `QDRANT_API_KEY` | none | Read-write Database API key for the cluster. |
| `QDRANT_COLLECTION` | `sre-incidents` | Collection holding resolved incidents. |
| `HF_INCIDENTS_BUCKET` | none | Optional `username/bucket` to also archive incidents as `incidents/<id>.json`. Empty disables it. |

## License

[MIT](LICENSE) © 2026 Arkadiusz Borucki
