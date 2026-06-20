"""HF Storage Bucket archive for resolved incidents (optional, best-effort).

Companion to the Qdrant store. Qdrant is the search index; this is a durable,
S3-like copy on a Hugging Face Storage Bucket (Xet-backed object storage). Each
incident is written as one JSON object at `incidents/<id>.json`.

Enabled only when HF_INCIDENTS_BUCKET is set (e.g. "username/sre-agent-incidents").
If unset, archiving is skipped and the agent runs on Qdrant alone. Failures here
never break a save: Qdrant is the source of truth for search.

Note: buckets are non-versioned and mutable, so re-saving the same id overwrites
the previous object (no history). Auth uses HF_TOKEN (a write-scoped token for
the target namespace), read from the environment by huggingface_hub.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

BUCKET = os.getenv("HF_INCIDENTS_BUCKET")  # "username/sre-agent-incidents" or None


def enabled() -> bool:
    return bool(BUCKET and os.getenv("HF_TOKEN"))


@lru_cache(maxsize=1)
def _ensure_bucket() -> str:
    """Create the bucket if needed (private). Returns the bucket id."""
    from huggingface_hub import create_bucket

    create_bucket(BUCKET, private=True, exist_ok=True)
    return BUCKET


def archive_incident(incident_id: str, payload: dict) -> str | None:
    """Write one incident as incidents/<id>.json into the bucket. Returns the
    hf:// URI, or None if archiving is disabled. Raises only on a real upload
    failure (the caller wraps this so a failure degrades to a warning)."""
    if not enabled():
        return None

    from huggingface_hub import batch_bucket_files

    record = {"id": incident_id, **payload}
    blob = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    path_in_bucket = f"incidents/{incident_id}.json"
    # add=[(source, dest)]: source may be bytes or a local path. Non-transactional.
    batch_bucket_files(_ensure_bucket(), add=[(blob, path_in_bucket)])
    return f"hf://buckets/{BUCKET}/{path_in_bucket}"
