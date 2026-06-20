"""One-time setup: create the `sre-incidents` Qdrant collection.

Hybrid search over past, resolved incidents so the agent can recall a prior
root cause before re-investigating from scratch. Two named vectors, both
computed server-side by Qdrant Cloud Inference (the agent never embeds locally):

  - "e5"   : dense, intfloat/multilingual-e5-small (384-dim, cosine), matches
             symptom descriptions by meaning and is multi-language, so a
             paraphrased symptom still finds the same past failure.
  - "bm25" : sparse, qdrant/bm25 (IDF), exact keyword matches (error codes,
             namespace names, signals like OOMKilled / exit_code=137).

Run once:
    pip install -r requirements.txt
    cp .env.example .env   # fill QDRANT_URL + QDRANT_API_KEY
    python setup_qdrant.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = os.getenv("QDRANT_COLLECTION", "sre-incidents")

# Cloud Inference model identifiers. If the API rejects a name, copy the exact
# string shown in your cluster's Inference panel — that is the ground truth.
E5_MODEL = "intfloat/multilingual-e5-small"  # dense, 384-dim, 512-token context
BM25_MODEL = "qdrant/bm25"                    # sparse


def main() -> None:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        sys.exit("Set QDRANT_URL and QDRANT_API_KEY in .env (copy .env.example).")

    # cloud_inference=True makes models.Document(text=...) embed server-side.
    client = QdrantClient(url=url, api_key=api_key, cloud_inference=True)

    if client.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' already exists — nothing to do.")
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "e5": models.VectorParams(size=384, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            # IDF is computed server-side; recommended for BM25.
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    print(f"Created collection '{COLLECTION}' (e5 dense + bm25 sparse, hybrid).")


if __name__ == "__main__":
    main()
