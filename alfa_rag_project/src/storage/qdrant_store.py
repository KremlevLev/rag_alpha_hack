"""
Enterprise vector store with Qdrant hybrid search.

Supports:
  - Dense vector indexing (from Voyage/Cohere/BGE)
  - Sparse vector indexing (BM25 via Qdrant built-in sparse)
  - Hybrid search (dense + sparse with fusion)
  - Batch upload with retry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """A single result from hybrid search."""

    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class QdrantStore:
    """
    Qdrant hybrid vector store.

    Usage:
        store = QdrantStore(url="http://localhost:6333", collection="my_collection")
        await store.create_collection(vector_size=1024)
        await store.add_chunks(chunks, embeddings)
        results = await store.hybrid_search(query_vector, query_text, top_k=50)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        grpc_port: int = 6334,
        api_key: Optional[str] = None,
        collection_name: str = "enterprise_rag",
        prefer_grpc: bool = False,
    ) -> None:
        self._collection = collection_name
        self._client = QdrantClient(
            url=url,
            port=6333,
            grpc_port=grpc_port,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=30,
        )

    async def create_collection(self, vector_size: int, distance: str = "Cosine") -> None:
        """Create the Qdrant collection with dense + sparse vector config."""
        collections = self._client.get_collections()
        existing = {c.name for c in collections.collections}
        if self._collection in existing:
            logger.info("Collection %s already exists, skipping creation", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance[distance.upper()],
                on_disk=False,
            ),
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        logger.info(
            "Created collection %s (vector_size=%d, distance=%s)",
            self._collection,
            vector_size,
            distance,
        )

    async def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadata_list: Optional[List[dict]] = None,
    ) -> None:
        """Upload chunks with dense vectors and sparse BM25 tokens."""
        if metadata_list is None:
            metadata_list = [{} for _ in chunk_ids]

        points: List[models.PointStruct] = []
        for idx, (cid, text, emb) in enumerate(zip(chunk_ids, texts, embeddings)):
            sparse = self._text_to_sparse(text)
            point = models.PointStruct(
                id=idx,
                vector={
                    "": emb,
                    "bm25": models.SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload={
                    "chunk_id": cid,
                    "text": text,
                    **metadata_list[idx],
                },
            )
            points.append(point)

        self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        logger.info("Uploaded %d chunks to Qdrant", len(points))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    async def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 50,
    ) -> List[HybridSearchResult]:
        """Hybrid search: dense vector + sparse BM25 with RRF fusion."""

        sparse = self._text_to_sparse(query_text)

        results, _ = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=None,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    limit=top_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    limit=top_k,
                ),
            ],
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        output: List[HybridSearchResult] = []
        for point in results.points:
            output.append(
                HybridSearchResult(
                    chunk_id=point.payload.get("chunk_id", str(point.id)),
                    text=point.payload.get("text", ""),
                    score=point.score,
                    metadata={k: v for k, v in point.payload.items() if k not in ("chunk_id", "text")},
                )
            )
        return output

    async def delete_collection(self) -> None:
        """Drop the collection."""
        self._client.delete_collection(collection_name=self._collection)
        logger.info("Deleted collection %s", self._collection)

    async def count(self) -> int:
        """Return the number of points in the collection."""
        result = self._client.count(collection_name=self._collection)
        return result.count

    @staticmethod
    def _text_to_sparse(text: str) -> models.SparseVector:
        """Tokenize text into sparse BM25 vector.

        Uses simple whitespace tokenization; for production, use
        a proper tokenizer (razdel, spacy, etc.).
        """
        import re

        tokens = re.findall(r"\w{2,}", text.lower())
        # Aggregate: unique token → frequency (simplified; Qdrant applies IDF)
        from collections import Counter

        freq = Counter(tokens)
        return models.SparseVector(
            indices=[hash(t) % (2**31 - 1) for t in freq.keys()],
            values=list(freq.values()),
        )