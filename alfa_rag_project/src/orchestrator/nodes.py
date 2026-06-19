"""
LangGraph computation nodes for the Enterprise RAG pipeline.

Each node is an async function that takes `AgentState` and returns
a dict with field updates to merge into the state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from config import TOP_K_RETRIEVAL, TOP_K_BM25
from orchestrator.state import AgentState, RetrievalChunk

logger = logging.getLogger(__name__)


async def retrieve_node(state: AgentState, retriever: Any = None) -> Dict[str, Any]:
    """
    Hybrid search: retrieve chunks from FAISS + BM25.

    Uses the existing Retriever instance from retriever.py.
    Returns top-k candidates merged via RRF.
    """
    query = state.rewritten_query or state.query
    logger.info("retrieve_node: query='%s' (rewritten=%s)", query[:60], state.rewritten_query is not None)

    if retriever is None:
        raise RuntimeError("retrieve_node requires a 'retriever' argument (Retriever instance)")

    try:
        raw_chunks = retriever.retrieve(query)
    except Exception as exc:
        logger.error("retrieval failed: %s", exc)
        return {"errors": [f"retrieval failed: {exc}"], "retrieved_chunks": []}

    # raw_chunks are (parent_id, text, score) from Retriever.retrieve()
    chunks: list[RetrievalChunk] = []
    for cid, text, score in raw_chunks:
        chunks.append(RetrievalChunk(chunk_id=str(cid), text=text, dense_score=score))

    return {"retrieved_chunks": chunks}


async def rerank_node(state: AgentState, reranker: Any = None) -> Dict[str, Any]:
    """
    Rerank retrieved chunks using the cross-encoder / Cohere Rerank.

    If CohereReranker is provided, use it; otherwise use the existing
    BGE cross-encoder from retriever.py.
    """
    if not state.retrieved_chunks:
        logger.warning("rerank_node: no chunks to rerank")
        return {"reranked_chunks": []}

    query = state.rewritten_query or state.query
    logger.info("rerank_node: reranking %d chunks", len(state.retrieved_chunks))

    try:
        if reranker is not None and hasattr(reranker, "rerank"):
            # CohereReranker style
            documents = [(c.chunk_id, c.text) for c in state.retrieved_chunks]
            reranked = await reranker.rerank(query, documents, top_k=5)
            chunks = [
                RetrievalChunk(chunk_id=cid, text=text, rerank_score=score)
                for cid, text, score in reranked
            ]
        else:
            # BGE cross-encoder style — scored in retriever.retrieve()
            chunks = sorted(
                state.retrieved_chunks,
                key=lambda c: c.dense_score,
                reverse=True,
            )[:5]
    except Exception as exc:
        logger.error("rerank failed: %s", exc)
        chunks = state.retrieved_chunks[:5]

    return {"reranked_chunks": chunks}


async def quality_gate_node(state: AgentState) -> Dict[str, Any]:
    """
    Quality gate: if the best rerank score is below threshold, trigger rewrite.

    Threshold can be configured. Default: 0.65 for Cohere, 0.05 for BGE local.
    """
    if not state.reranked_chunks:
        logger.info("quality_gate: no reranked chunks — need rewrite")
        return {"rewrite_count": state.rewrite_count + 1, "context": ""}

    best_score = max(c.rerank_score or c.dense_score for c in state.reranked_chunks)
    logger.info("quality_gate: best_score=%.4f", best_score)

    threshold = 0.65 if state.reranked_chunks[0].rerank_score is not None else 0.05
    if best_score < threshold and state.rewrite_count < 3:
        logger.info("quality_gate: score %.4f < %.2f — rewriting query", best_score, threshold)
        return {"rewrite_count": state.rewrite_count + 1, "context": ""}

    # Build context from reranked chunks
    context_parts: list[str] = []
    for c in state.reranked_chunks:
        context_parts.append(c.text)
    context = "\n\n".join(context_parts)

    return {"context": context}


async def rewrite_node(state: AgentState, llm_client: Any = None) -> Dict[str, Any]:
    """
    Rewrite the query to improve retrieval quality.

    Uses a lightweight LLM (e.g., GPT-4o-mini or OpenRouter) to reformulate.
    Falls back to prepending "Уточни: " if no LLM client is available.
    """
    if state.rewrite_count > 3:
        return {"errors": [f"Max rewrites ({state.rewrite_count}) exceeded"]}

    if llm_client is not None and hasattr(llm_client, "generate"):
        prompt = (
            f"Переформулируй следующий запрос для поиска в банковской базе знаний. "
            f"Сохрани суть, добавь ключевые термины.\n\nЗапрос: {state.query}\n\n"
            f"Переформулированный запрос:"
        )
        try:
            rewritten = await llm_client.generate(prompt, "")
            if rewritten.strip():
                logger.info("rewrite_node: '%s' -> '%s'", state.query[:40], rewritten[:60])
                return {"rewritten_query": rewritten.strip()}
        except Exception as exc:
            logger.warning("rewrite_node LLM failed: %s", exc)

    # Fallback
    rewritten = f"Уточни: {state.query}"
    return {"rewritten_query": rewritten}


async def generate_node(state: AgentState, generator: Any = None) -> Dict[str, Any]:
    """
    Generate a final answer from query and context.

    Uses the provided generator (Claude, OpenRouter, or existing KaggleGenerator).
    """
    query = state.rewritten_query or state.query
    context = state.context or ""

    if not context:
        logger.warning("generate_node: empty context — answer may be low quality")
    logger.info("generate_node: query='%s', context_len=%d", query[:60], len(context))

    if generator is None:
        raise RuntimeError("generate_node requires a 'generator' argument")

    try:
        if hasattr(generator, "generate") and not hasattr(generator, "__call__"):
            raw = generator.generate(query, context)
        else:
            raw = await generator.generate(query, context) if hasattr(generator, "__call__") else generator(query, context)
    except Exception as exc:
        logger.error("generation failed: %s", exc)
        return {"errors": [f"generation failed: {exc}"], "raw_answer": ""}

    return {"raw_answer": raw.strip()}


async def validate_node(state: AgentState, validator: Any = None) -> Dict[str, Any]:
    """
    Validate the generated answer for hallucination / emptiness.

    If validation fails, increment rewrite_count to trigger re-retrieval.
    """
    if not state.raw_answer:
        logger.warning("validate_node: empty answer")
        return {"errors": ["empty answer"], "rewrite_count": state.rewrite_count + 1}

    # Basic emptiness check
    if len(state.raw_answer.split()) < 3:
        logger.warning("validate_node: answer too short (%d words)", len(state.raw_answer.split()))
        return {"errors": ["answer too short"], "rewrite_count": state.rewrite_count + 1}

    if validator is not None and hasattr(validator, "validate"):
        try:
            score, feedback = await validator.validate(state.query, state.raw_answer, state.context)
            logger.info("validate_node: score=%.4f, feedback='%s'", score, feedback[:60])
            if score < 0.5:
                return {"errors": [f"validation failed: {feedback}"], "rewrite_count": state.rewrite_count + 1}
        except Exception as exc:
            logger.warning("validate_node validator call failed: %s", exc)

    return {}


async def finalize_node(state: AgentState) -> Dict[str, Any]:
    """
    Finalize the answer: post-process and set final_answer.
    """
    answer = state.raw_answer
    if not answer and state.reranked_chunks:
        # Fallback: first sentence of best chunk
        answer = state.reranked_chunks[0].text.split(".")[0] + "."
    elif not answer:
        answer = "Извините, не удалось найти ответ."

    return {"final_answer": answer.strip()}