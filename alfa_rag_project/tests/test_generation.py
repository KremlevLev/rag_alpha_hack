"""
Tests for the LLM generation clients.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from generation import LocalClient, create_llm_client


def test_create_local_client() -> None:
    client = create_llm_client("local", model_name="Vikhrmodels/Vikhr-Llama-3.2-1B-instruct")
    assert isinstance(client, LocalClient)


def test_anthropic_client_requires_key() -> None:
    from generation.llm_client import AnthropicClient

    with pytest.raises(TypeError):  # missing required api_key argument
        AnthropicClient()