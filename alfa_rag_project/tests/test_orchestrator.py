"""
Tests for the LangGraph orchestration pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio

from orchestrator.state import AgentState, RetrievalChunk
from orchestrator.nodes import quality_gate_node, finalize_node


def test_agent_state_defaults() -> None:
    state = AgentState(query="Как заблокировать карту?")
    assert state.query == "Как заблокировать карту?"
    assert state.rewritten_query is None
    assert state.rewrite_count == 0
    assert state.retrieved_chunks == []
    assert state.reranked_chunks == []
    assert state.context == ""
    assert state.final_answer == ""


def test_quality_gate_accepts_good_rerank_score() -> None:
    async def _test() -> None:
        state = AgentState(
            query="test",
            reranked_chunks=[
                RetrievalChunk(chunk_id="1", text="Some text", rerank_score=0.85),
            ],
        )
        result = await quality_gate_node(state)
        assert "rewrite_count" not in result or result.get("rewrite_count", 0) == 0
        assert result["context"] != ""

    asyncio.run(_test())


def test_quality_gate_triggers_rewrite_on_low_score() -> None:
    async def _test() -> None:
        state = AgentState(
            query="test",
            reranked_chunks=[
                RetrievalChunk(chunk_id="1", text="Some text", rerank_score=0.12),
            ],
            rewrite_count=0,
        )
        result = await quality_gate_node(state)
        assert result.get("rewrite_count", 0) == 1
        assert result.get("context") == ""

    asyncio.run(_test())


def test_finalize_falls_back_to_chunk_text() -> None:
    async def _test() -> None:
        state = AgentState(
            query="test",
            raw_answer="",
            reranked_chunks=[
                RetrievalChunk(chunk_id="1", text="Это ответ. Ещё предложение."),
            ],
        )
        result = await finalize_node(state)
        assert result["final_answer"] == "Это ответ."

    asyncio.run(_test())


def test_finalize_default_message_on_empty() -> None:
    async def _test() -> None:
        state = AgentState(query="test")
        result = await finalize_node(state)
        assert result["final_answer"] == "Извините, не удалось найти ответ."

    asyncio.run(_test())