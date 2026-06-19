"""
Tests for observability tracing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from observability import NullTracer, create_tracer


def test_null_tracer_returns_trace_id() -> None:
    tracer = NullTracer()
    trace_id = tracer.start_trace("test")
    assert len(trace_id) == 16
    tracer.end_trace(trace_id)


def test_null_tracer_context_manager() -> None:
    tracer = NullTracer()
    with tracer.trace("test-context") as trace_id:
        assert len(trace_id) == 16


def test_create_tracer_default_is_null() -> None:
    tracer = create_tracer("null")
    assert isinstance(tracer, NullTracer)