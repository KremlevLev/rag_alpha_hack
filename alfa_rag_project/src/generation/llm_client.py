"""
Async LLM clients for generation.

Supports:
  - Anthropic Claude 3.5 Sonnet (via API)
  - OpenRouter (unified API for Claude, GPT-4o, Llama, etc.)
  - Local transformers pipeline (fallback)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — банковский AI-ассистент Альфа-Банка.
Отвечай только на вопрос клиента. Используй контекст, но не копируй его дословно.
Формат ответа: только сам ответ. Не возвращай служебные строки."""


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, query: str, context: str) -> str:
        ...


# ── Anthropic Claude ──────────────────────────────────────────


class AnthropicClient(LLMClient):
    """Anthropic Claude 3.5 Sonnet client."""

    BASE_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    async def generate(self, query: str, context: str) -> str:
        user_msg = f"Контекст:\n{context}\n\nВопрос: {query}" if context else query
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self.BASE_URL,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "max_tokens": 1024,
                    "temperature": 0.1,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_msg}],
                },
            )
            if resp.status_code != 200:
                logger.error("Anthropic error %d: %s", resp.status_code, resp.text[:200])
                resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"].strip()


# ── OpenRouter ────────────────────────────────────────────────


class OpenRouterClient(LLMClient):
    """OpenRouter unified API client."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet") -> None:
        self._api_key = api_key
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    async def generate(self, query: str, context: str) -> str:
        user_msg = f"Контекст:\n{context}\n\nВопрос: {query}" if context else query
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 1024,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
            if resp.status_code != 200:
                logger.error("OpenRouter error %d: %s", resp.status_code, resp.text[:200])
                resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()


# ── Local transformers fallback ───────────────────────────────


class LocalClient(LLMClient):
    """Local transformers pipeline (Hugging Face)."""

    def __init__(self, model_name: str = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct") -> None:
        self._model_name = model_name
        self._pipeline = None

    async def generate(self, query: str, context: str) -> str:
        pipe = await self._get_pipeline()
        user_msg = f"Контекст:\n{context}\n\nВопрос: {query}" if context else query
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = pipe(
            prompt,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            return_full_text=False,
        )
        return result[0]["generated_text"].strip()

    async def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline

            logger.info("Loading local model: %s", self._model_name)
            self._pipeline = hf_pipeline("text-generation", model=self._model_name, device_map="auto")
        return self._pipeline


# ── Factory ───────────────────────────────────────────────────


def create_llm_client(provider: str = "openrouter", **kwargs) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "openrouter":
        return OpenRouterClient(**kwargs)
    else:
        return LocalClient(**kwargs)