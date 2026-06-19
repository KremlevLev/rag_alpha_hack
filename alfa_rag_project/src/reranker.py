"""
Enterprise reranker — Cohere Rerank v3 with BGE fallback.

Async, retry-safe. Replaces or augments the existing BGE cross-encoder reranker.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from retriever import clean_chunk_text

logger = logging.getLogger(__name__)

COHERE_API_KEY: Optional[str] = None
COHERE_RERANK_MODEL: str = "rerank-v3.5"


def set_cohere_key(key: str) -> None:
    global COHERE_API_KEY
    COHERE_API_KEY = key


# ── Cohere Rerank ──────────────────────────────────────────────


class CohereReranker:
    """Cohere Rerank v3 client."""

    BASE_URL = "https://api.cohere.com/v2/rerank"

    def __init__(self, api_key: Optional[str] = None, model: str = COHERE_RERANK_MODEL) -> None:
        self._api_key = api_key or COHERE_API_KEY
        if not self._api_key:
            raise ValueError("Cohere API key not set.")
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    async def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str]],
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query.
            documents: List of (doc_id, text) pairs.
            top_k: Number of top results to return.

        Returns:
            List of (doc_id, text, score) sorted by relevance descending.
        """
        cleaned_docs = [
            clean_chunk_text(text) or text
            for _, text in documents
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "query": query,
                    "documents": cleaned_docs,
                    "top_n": top_k,
                    "return_documents": True,
                },
            )

            if resp.status_code != 200:
                logger.error("Cohere Rerank error %d: %s", resp.status_code, resp.text[:200])
                resp.raise_for_status()

            data = resp.json()

        results: List[Tuple[str, str, float]] = []
        for item in data.get("results", []):
            idx = item["index"]
            doc_id, _ = documents[idx]
            text = cleaned_docs[idx]
            score = item.get("relevance_score", 0.0)
            results.append((doc_id, text, score))

        return results


# ── Factory ─────────────────────────────────────────────────────


def create_reranker(provider: str = "cohere", **kwargs):
    """Factory: create reranker by provider name.

    Args:
        provider: 'cohere' or 'bge'.
        **kwargs: Passed to constructor.

    Returns:
        CohereReranker if 'cohere', else the existing BGE reranker from retriever.
    """
    if provider == "cohere":
        return CohereReranker(**kwargs)
    else:
        from sentence_transformers import CrossEncoder
        import torch

        rerank_device = (
            "cuda:1"
            if torch.cuda.device_count() >= 2
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        model_name = kwargs.get("model", "BAAI/bge-reranker-v2-m3")
        return CrossEncoder(model_name, device=rerank_device)