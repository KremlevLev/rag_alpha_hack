"""
LangGraph StateGraph assembly for the Enterprise RAG pipeline.

Builds the full RAG graph: retrieve → rerank → quality gate →
(rewrite → retrieve ... ) → generate → validate → finalize.

Edge cases:
  - If quality check fails → reroute to rewrite_node
  - If validation fails → reroute to rewrite_node (up to 3 retries)
  - If max retries exceeded → go directly to finalize
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from orchestrator.nodes import (
    finalize_node,
    generate_node,
    quality_gate_node,
    rerank_node,
    retrieve_node,
    rewrite_node,
    validate_node,
)
from orchestrator.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_quality(state: AgentState) -> Literal["rewrite", "generate"]:
    """If quality gate failed and rewrites remain → rewrite. Otherwise → generate."""
    if not state.context and state.rewrite_count <= 3:
        return "rewrite"
    return "generate"


def _route_after_validate(state: AgentState) -> Literal["rewrite", "finalize"]:
    """If validation failed and rewrites remain → rewrite. Otherwise → finalize."""
    if state.errors and state.rewrite_count <= 3:
        return "rewrite"
    return "finalize"


def _route_after_rewrite(state: AgentState) -> Literal["retrieve", "finalize"]:
    """If max rewrites exceeded → finalize with best effort. Otherwise → retrieve."""
    if state.rewrite_count > 3:
        return "finalize"
    return "retrieve"


def build_rag_graph(
    retriever: Any = None,
    reranker: Any = None,
    generator: Any = None,
    validator: Any = None,
    llm_client: Any = None,
) -> StateGraph:
    """
    Assemble the LangGraph StateGraph for the RAG pipeline.

    Args:
        retriever: Retriever instance (from retriever.py).
        reranker: CohereReranker or None (uses BGE fallback).
        generator: Generator instance (KaggleGenerator, VLLMGenerator, or async LLM client).
        validator: Optional validator for hallucination check.
        llm_client: Optional LLM client for query rewriting.

    Returns:
        Compiled StateGraph ready for invocation.
    """

    workflow = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────
    workflow.add_node("retrieve", lambda s: retrieve_node(s, retriever=retriever))
    workflow.add_node("rerank", lambda s: rerank_node(s, reranker=reranker))
    workflow.add_node("quality_gate", quality_gate_node)
    workflow.add_node("rewrite", lambda s: rewrite_node(s, llm_client=llm_client))
    workflow.add_node("generate", lambda s: generate_node(s, generator=generator))
    workflow.add_node("validate", lambda s: validate_node(s, validator=validator))
    workflow.add_node("finalize", finalize_node)

    # ── Edges ──────────────────────────────────────────────────
    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "quality_gate")

    workflow.add_conditional_edges(
        "quality_gate",
        _route_after_quality,
        {"rewrite": "rewrite", "generate": "generate"},
    )

    workflow.add_conditional_edges(
        "rewrite",
        _route_after_rewrite,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )

    workflow.add_edge("generate", "validate")

    workflow.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"rewrite": "rewrite", "finalize": "finalize"},
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


async def run_rag_pipeline(
    query: str,
    retriever: Any = None,
    reranker: Any = None,
    generator: Any = None,
    validator: Any = None,
    llm_client: Any = None,
) -> AgentState:
    """
    Convenience function: build graph and run a single query.

    Returns the final AgentState with final_answer populated.
    """
    graph = build_rag_graph(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        validator=validator,
        llm_client=llm_client,
    )

    initial_state = AgentState(query=query)
    result = await graph.ainvoke(initial_state)
    return AgentState(**result) if isinstance(result, dict) else result