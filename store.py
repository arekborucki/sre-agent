"""Incident memory: store resolved incidents and recall similar ones.

This is the *only* place that knows about Qdrant. The agent and its tools talk
to `save_incident` / `search_incidents`, so the backend can be swapped without
touching them.

Both vectors are computed server-side by Qdrant Cloud Inference (the client is
created with cloud_inference=True), so we only ever send text — no local
embedding model, no GPU:
  - "e5"   dense  : intfloat/multilingual-e5-small  (meaning, PL+EN)
  - "bm25" sparse : qdrant/bm25                      (exact tokens)
Queries fuse both lists with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from qdrant_client import QdrantClient, models

COLLECTION = os.getenv("QDRANT_COLLECTION", "sre-incidents")
E5_MODEL = "intfloat/multilingual-e5-small"
BM25_MODEL = "qdrant/bm25"


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in the environment.")
    return QdrantClient(url=url, api_key=api_key, cloud_inference=True)


def _embedding_text(title: str, symptom: str, signals: list[str] | None) -> str:
    """The text we embed/index. Combining title + symptom + signals gives both
    the dense and sparse vectors more to match on than the symptom alone."""
    parts = [title, symptom]
    if signals:
        parts.append(" ".join(signals))
    return "\n".join(p for p in parts if p)


def save_incident(
    title: str,
    symptom: str,
    root_cause: str,
    fix: str,
    environment: dict | None = None,
    signals: list[str] | None = None,
    commands_run: list[str] | None = None,
) -> str:
    """Store a resolved incident. Returns the new point id."""
    text = _embedding_text(title, symptom, signals)
    point_id = uuid.uuid4().hex
    _client().upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "e5": models.Document(text=text, model=E5_MODEL),
                    "bm25": models.Document(text=text, model=BM25_MODEL),
                },
                payload={
                    "title": title,
                    "symptom": symptom,
                    "root_cause": root_cause,
                    "fix": fix,
                    "environment": environment or {},
                    "signals": signals or [],
                    "commands_run": commands_run or [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )
    return point_id


def search_incidents(symptom: str, top_k: int = 5) -> list[dict]:
    """Hybrid search (dense + BM25, RRF-fused) for past incidents resembling
    `symptom`. Returns the top matches with their root cause and fix."""
    results = _client().query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=symptom, model=E5_MODEL),
                using="e5",
                limit=max(20, top_k * 4),
            ),
            models.Prefetch(
                query=models.Document(text=symptom, model=BM25_MODEL),
                using="bm25",
                limit=max(20, top_k * 4),
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    out = []
    for p in results.points:
        payload = p.payload or {}
        out.append(
            {
                "score": round(p.score, 4),
                "title": payload.get("title"),
                "symptom": payload.get("symptom"),
                "root_cause": payload.get("root_cause"),
                "fix": payload.get("fix"),
                "environment": payload.get("environment"),
                "signals": payload.get("signals"),
                "timestamp": payload.get("timestamp"),
            }
        )
    return out
