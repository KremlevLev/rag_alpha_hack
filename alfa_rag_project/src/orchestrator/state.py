"""
LangGraph AgentState for the RAG pipeline.

Carries query, rewritten queries, retrieved/reranked chunks,
generated answers, and error logs through the graph nodes.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RetrievalChunk(BaseModel):
    """A chunk returned from hybrid search or reranker."""

    chunk_id: str = ""
    text: str = ""
    dense_score: float = 0.0
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None


class AgentState(BaseModel):
    """
    State passed between LangGraph nodes.

    Each node reads from and writes to this state.
    """

    query: str = Field(description="Original user query")
    rewritten_query: Optional[str] = Field(default=None, description="Reformulated query after rewrite")
    rewrite_count: int = Field(default=0, ge=0, le=3)
    retrieved_chunks: List[RetrievalChunk] = Field(default_factory=list)
    reranked_chunks: List[RetrievalChunk] = Field(default_factory=list)
    context: str = Field(default="")
    raw_answer: str = Field(default="")
    final_answer: str = Field(default="")
    errors: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = Field(default=None)