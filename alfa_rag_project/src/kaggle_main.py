"""
Kaggle-optimized RAG pipeline.
Uses Hugging Face Inference API or local transformers for open-source LLMs.
Supports 2x T4 GPU for faster inference.
"""

import gc
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# CUDA OOM mitigation: expandable segments for memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import razdel
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import is_flash_attn_2_available

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    INDEX_PATH,
    QUESTIONS_CSV,
    SUBMISSION_CSV,
    WEBSITES_CSV,
    MAX_SENTENCES,
    MAX_RESPONSE_WORDS,
    MAX_RESPONSE_CHARS,
    TEMPERATURE,
    LLM_TIMEOUT,
)
from chunker import chunk_all_websites
from generator import extract_answer_from_context
from indexer import build_and_save_index, load_index
from retriever import create_retriever

# ─────────────────────────────────────────────
# Логирование
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Kaggle-оптимизированный генератор
# ─────────────────────────────────────────────

# Open-source модели, совместимые с Kaggle
KAGGLE_MODELS = {
    "vikhr-1b-finetuned": "lirex111/vikhrllama1B_AlfaBank",  # Fine-tuned Vikhr-1B for Alfa-Bank (RECOMMENDED)
    "vikhr-1b": "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct",  # Base 1B model
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# Flag for int8 quantization (speeds up inference on T4 GPU)
USE_INT8: bool = False  # Disabled - was slower on Kaggle T4

import re


# ─────────────────────────────────────────────
# Pipeline version (менять при правке промпта/чанкеров/пост-процессинга)
# ─────────────────────────────────────────────
PIPELINE_VERSION: str = "v2-adaptive-len"


# System prompt для русскоязычных моделей с few-shot примерами
SYSTEM_PROMPT = """Ты — банковский AI-ассистент Альфа-Банка. Отвечай на вопрос ПОЛНО и фактически точно, опираясь ТОЛЬКО на предоставленный контекст.

ПРАВИЛА:
1. Используй все релевантные факты из контекста.
2. Если прямого ответа в контексте нет, всё равно выбери самое релевантное предложение из контекста и сформулируй ответ по нему.
3. Если ответ — это перечень шагов или вариантов, оформи его списком или через точку с запятой. Сохраняй ВСЕ пункты.
4. НЕ пиши вводных слов («Согласно фрагменту», «Таким образом», «В контексте указано»). Сразу давай суть.
5. Не здоровайся, не предлагай помощь, без рекламы.
6. Объём ответа — как в справке банка: обычно 2–5 предложений, при списках больше.

ПРИМЕРЫ:
Вопрос: Что такое БИК?
Контекст: БИК — банковский идентификационный код для перечисления средств.
Ответ: БИК — это банковский идентификационный код, используемый для перечисления средств.

Вопрос: Как получить карту?
Контекст: Карту можно получить доставкой или в офисе. После получения нужно подписать договор и активировать карту в приложении: выберите карту → Активация → введите код из SMS → задайте пин-код.
Ответ: Получить карту можно доставкой или в офисе. После получения подпишите договор и активируйте карту в приложении: выберите карту на главном экране, нажмите «Активация», введите код из SMS и задайте пин-код.

Вопрос: Как узнать номер счёта?
Контекст: Номер счёта отображается в мобильном приложении банка на вкладке «Мои счета». Также можно позвонить в поддержку.
Ответ: Номер счёта отображается в мобильном приложении банка на вкладке «Мои счета». Для уточнения можно позвонить в поддержку.
""".strip()


# ─────────────────────────────────────────────
# Стрипалка «воды» из LLM-ответа
# ─────────────────────────────────────────────

_PREAMBLE_RE = re.compile(
    r"^\s*(?:согласно\s+(?:фрагмент[ауые]*\s*\d*|контекст[ауе]*|предоставленны[мх][^,.:]*)[,:]?\s*"
    r"|в\s+фрагмент[еах]*\s*\d*\s*(?:указано|сказано|говорится)[,:]?\s*"
    r"|таким образом[,:]?\s*"
    r"|исходя из (?:контекста|вышесказанного)[,:]?\s*"
    r"|ответ[:：]\s*)",
    flags=re.IGNORECASE | re.UNICODE,
)


def strip_preamble(text: str) -> str:
    """Убирает мета-вводные клише, разбавляющие ответ."""
    prev = None
    while prev != text:  # многократно, если клише вложены
        prev = text
        text = _PREAMBLE_RE.sub("", text).lstrip(" «\"—-:")
    # убрать остаточные «(Фрагмент 3)» в конце
    text = re.sub(r"\s*\(?\s*фрагмент\s*\d+\s*\)?\.?\s*$", "", text, flags=re.IGNORECASE).strip()
    return text


def truncate_to_chars(text: str, max_chars: int) -> str:
    """
    Truncate text to maximum character count.

    This is a SAFETY limit to prevent 3x length penalty.

    Args:
        text: Input text
        max_chars: Maximum number of characters

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    for punct in [".", "!", "?", "»"]:
        last_punct = truncated.rfind(punct)
        if last_punct > max_chars * 0.3:
            return truncated[:last_punct + 1]

    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:
        return truncated[:last_space]

    return truncated


def truncate_to_sentences(text: str, max_sentences: int) -> str:
    """
    Truncate text to maximum number of sentences.

    This is the PRIMARY truncation for BERT-Recall-L compliance.

    Args:
        text: Input text
        max_sentences: Maximum number of sentences

    Returns:
        Truncated text
    """
    sentences = [s.text.strip() for s in razdel.sentenize(text) if s.text.strip()]

    if len(sentences) <= max_sentences:
        return text

    return " ".join(sentences[:max_sentences])


class KaggleGenerator:
    """
    Hugging Face генератор для Kaggle.
    Автоматически использует доступные GPU.
    """

    def __init__(
        self,
        model_name: str = "lirex111/vikhrllama1B_AlfaBank",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize generator with Hugging Face model.

        Args:
            model_name: Hugging Face model identifier
            device_map: "auto" for multi-GPU, "cuda:0" for single GPU
            torch_dtype: float16 for T4 GPU (memory efficient)
        """
        self.model_name = model_name

        # Определяем количество доступных GPU
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} GPU(s) on Kaggle")

        if num_gpus >= 2:
            # Два T4: используем model parallelism
            device_map = "auto"  # Автоматическое распределение по GPU
            logger.info("Using multi-GPU (2x T4) with device_map='auto'")
        elif num_gpus == 1:
            device_map = "cuda:0"
            logger.info("Using single GPU (T4)")
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            logger.warning("No GPU available, falling back to CPU (slow)")

        # Загружаем модель
        logger.info(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        # Устанавливаем pad_token если нужно
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "use_cache": True,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # Создаем pipeline — НЕ передаём device_map/torch_dtype повторно,
        # модель уже размещена на GPU через AutoModelForCausalLM.
        # Повторная передача device_map в pipeline может вызвать двойное размещение / OOM на 2×T4.
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        logger.info("Model loaded successfully")

    def generate(self, query: str, context: str) -> str:
        """
        Generate answer for query using context.

        Args:
            query: User question
            context: Retrieved context from retriever

        Returns:
            Generated answer (truncated to max sentences and chars)
        """
        # Даже пустой контекст должен вернуть что-то из fallback
        if not context:
            return extract_answer_from_context(query, "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Вопрос: {query}\n\nКонтекст:\n{context}\n\nОтветь кратко на основе контекста."},
        ]

        try:
            # Формируем промпт в зависимости от модели
            if "Qwen" in self.model_name:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif "Llama" in self.model_name or "Vikhr" in self.model_name or "vikhr" in self.model_name:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Универсальный формат
                prompt = f"{SYSTEM_PROMPT}\n\nВопрос: {query}\n\nКонтекст:\n{context}\n\nОтвет:"

            # CUDA OOM mitigation: отключаем градиенты во время инференса
            with torch.no_grad():
                outputs = self.pipe(
                    prompt,
                    max_new_tokens=320,
                    temperature=TEMPERATURE,
                    do_sample=TEMPERATURE > 0,
                    top_p=0.9 if TEMPERATURE > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,
                )

            answer = outputs[0]["generated_text"].strip()

            # Пост-обработка: чистим воду → режем по предложениям → safety по символам
            answer = strip_preamble(answer)               # FIX-G6
            answer = truncate_to_sentences(answer, MAX_SENTENCES)
            answer = truncate_to_chars(answer, MAX_RESPONSE_CHARS)

            # CUDA OOM mitigation
            gc.collect()
            torch.cuda.empty_cache()

            return answer

        except Exception as e:
            logger.error("Generation failed, using fallback: %s", e)
            return extract_answer_from_context(query, context)


# ─────────────────────────────────────────────
# VLLM генератор (опционально, ×5-10 throughput на T4)
# ─────────────────────────────────────────────

class VLLMGenerator:
    """
    Генератор на vLLM для максимального throughput на T4/L4.
    
    Использует PagedAttention + continuous batching.
    На Vikhr-1B на T4 даёт ×5-10 ускорение относительно pipeline.
    """

    def __init__(
        self,
        model_name: str = "lirex111/vikhrllama1B_AlfaBank",
        fast_gpu: bool = False,
    ):
        """
        Args:
            model_name: Hugging Face model identifier.
            fast_gpu: Режим для одной L4 (24GB) — больше памяти под vLLM
        """
        self.model_name = model_name

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        logger.info("Loading VLLM model: %s", model_name)
        # FIX: merged model имеет сломанный tokenizer_config (TokenizersBackend).
        # Поэтому для vLLM явно задаём tokenizer от base Vikhr.
        tokenizer_id = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct"
        gpu_memory_utilization = 0.80 if fast_gpu else 0.55
        self.llm = LLM(
            model=model_name,
            tokenizer=tokenizer_id,
            dtype="float16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,           # достаточно для контекста + ответа
        )

        # Стандартные параметры сэмплирования
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=0.9 if TEMPERATURE > 0 else 1.0,
            max_tokens=320,
            stop_token_ids=None,
        )
        logger.info("VLLM model loaded successfully: %s", model_name)

    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt from query and context using chat template."""
        # FIX: truncate context to avoid vLLM max_seq_len=4096 overflow
        # 4254 tokens warning means context is too long
        # Increased from 2500 to 3200 to preserve more context for quality
        if len(context) > 3200:
            context = context[:3200]
            # Try to end at sentence boundary
            last_punct = max(context.rfind("."), context.rfind("!"), context.rfind("?"), context.rfind("»"))
            if last_punct > 2200:
                context = context[:last_punct + 1]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Вопрос: {query}\n\nКонтекст:\n{context}\n\nОтветь кратко на основе контекста."},
        ]
        # vLLM требует токенайзер для apply_chat_template
        # FIX: merged model имеет сломанный tokenizer_config (TokenizersBackend)
        from transformers import AutoTokenizer
        tokenizer_id = "Vikhrmodels/Vikhr-Llama-3.2-1B-instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True,
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, query: str, context: str) -> str:
        """
        Generate answer using vLLM.

        Args:
            query: User question.
            context: Retrieved context from retriever.

        Returns:
            Generated answer (post-processed).
        """
        # Даже пустой контекст должен вернуть что-то из fallback
        if not context:
            return extract_answer_from_context(query, "")

        try:
            prompt = self._build_prompt(query, context)
            outputs = self.llm.generate([prompt], self.sampling_params)
            answer = outputs[0].outputs[0].text.strip()

            # Пост-обработка
            answer = strip_preamble(answer)
            answer = truncate_to_sentences(answer, MAX_SENTENCES)
            answer = truncate_to_chars(answer, MAX_RESPONSE_CHARS)

            # CUDA OOM mitigation: очистка после каждого вызова
            gc.collect()
            torch.cuda.empty_cache()

            return answer

        except Exception as e:
            logger.error("VLLM generation failed, using fallback: %s", e)
            return extract_answer_from_context(query, context)

    def generate_batch(
        self,
        queries: list[str],
        contexts: list[str],
    ) -> list[str]:
        """
        Generate answers for a batch of (query, context) pairs.
        vLLM processes them with continuous batching — максимальный throughput.

        Args:
            queries: List of user questions.
            contexts: List of retrieved contexts (same length).

        Returns:
            List of generated answers.
        """
        prompts: list[str] = []
        results: list[str] = [""] * len(queries)
        batch_map: list[int] = []  # which index in output → which index in input

        for i, (query, context) in enumerate(zip(queries, contexts)):
            if not context:
                answer = extract_answer_from_context(query, "")
                if not answer:
                    sentences = [s.strip() for s in re.split(r'[.!?»]+', context or "") if s.strip()]
                    answer = sentences[0] if sentences else query
                results[i] = answer
            else:
                prompts.append(self._build_prompt(query, context))
                batch_map.append(i)

        if not prompts:
            return results

        try:
            outputs = self.llm.generate(prompts, self.sampling_params)
            for j, output in enumerate(outputs):
                answer = output.outputs[0].text.strip()
                answer = strip_preamble(answer)
                answer = truncate_to_sentences(answer, MAX_SENTENCES)
                answer = truncate_to_chars(answer, MAX_RESPONSE_CHARS)
                results[batch_map[j]] = answer

            gc.collect()
            torch.cuda.empty_cache()
            return results

        except Exception as e:
            logger.error("VLLM batch generation failed: %s", e)
            # fallback for each prompt individually
            for i in range(len(prompts)):
                answer = extract_answer_from_context(queries[batch_map[i]], contexts[batch_map[i]])
                if not answer:
                    context = contexts[batch_map[i]]
                    sentences = [s.strip() for s in re.split(r'[.!?»]+', context or "") if s.strip()]
                    answer = sentences[0] if sentences else queries[batch_map[i]]
                results[batch_map[i]] = answer
            return results


# ─────────────────────────────────────────────
# Кеш ответов (тот же, что и в main.py)
# ─────────────────────────────────────────────

class AnswerCache:
    """Персистентный кеш ответов на диске (JSON) — с батч-сохранением."""

    def __init__(self, cache_path: Path):
        self._path = cache_path
        self._data: dict[str, dict] = {}
        self._dirty: bool = False
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("Cache loaded: %d entries", len(self._data))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Cache corrupted, starting fresh: %s", e)
                self._data = {}
        else:
            logger.info("No cache found, starting fresh")

    def save(self) -> None:
        """Сохраняет кеш на диск (только если были изменения)."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        self._dirty = False
        logger.debug("Cache saved: %d entries", len(self._data))

    @staticmethod
    def _make_key(query: str, model: str) -> str:
        raw = f"{PIPELINE_VERSION}|{query.strip()}|{model}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get(self, query: str, model: str) -> Optional[str]:
        key = self._make_key(query, model)
        entry = self._data.get(key)
        return entry["answer"] if entry else None

    def set(self, query: str, model: str, q_id: str, answer: str) -> None:
        """Добавляет запись в кеш (без немедленной записи на диск)."""
        key = self._make_key(query, model)
        self._data[key] = {
            "q_id": q_id,
            "query": query,
            "answer": answer,
            "model": model,
            "version": PIPELINE_VERSION,
        }
        self._dirty = True

    def __len__(self) -> int:
        return len(self._data)


# ─────────────────────────────────────────────
# Валидация ответов
# ─────────────────────────────────────────────

_STOP_WORDS = frozenset([
    "как", "что", "где", "когда", "почему", "зачем", "кто",
    "можно", "нужно", "надо", "это", "есть", "для", "при",
    "или", "если", "чтобы", "который", "которая", "которое",
    "в", "на", "с", "по", "из", "от", "до", "за", "к", "у",
    "я", "мы", "вы", "он", "она", "они",
])


def _extract_meaningful_words(text: str) -> set[str]:
    words = text.lower().split()
    return {
        w.strip(".,!?;:\"'()[]")
        for w in words
        if len(w) > 2 and w not in _STOP_WORDS
    }


def validate_answer(query: str, answer: str, min_overlap: int = 0) -> bool:
    """
    Очень мягкая проверка релевантности ответа вопросу.

    ВАЖНО: строгий word-overlap ломает RAG-качество — хорошие ответы
    часто не содержат дословных слов из вопроса, но семантически верны.
    Поэтому по умолчанию НЕ режем LLM-ответы.
    """
    if not answer or not answer.strip():
        return False

    # Никогда не режем осмысленный ответ только из-за отсутствия overlap
    if min_overlap <= 0:
        return True

    query_words = _extract_meaningful_words(query)
    answer_words = _extract_meaningful_words(answer)

    if not query_words:
        return True

    overlap = query_words & answer_words
    return len(overlap) >= min_overlap


# ─────────────────────────────────────────────
# Основной pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    build_index: bool = False,
    llm_model: str = "vikhr-1b-finetuned",
    cache_path: Path = Path("data/answer_cache.json"),
    validate_answers: bool = True,
    min_overlap: int = 1,
    use_vllm: bool = False,
    vllm_batch_size: int = 8,
    fast_gpu: bool = False,
) -> None:
    """
    Запускает полный RAG pipeline для Kaggle.

    Args:
        build_index: Строить индекс с нуля
        llm_model: Ключ модели из KAGGLE_MODELS
        cache_path: Путь к кешу
        validate_answers: Включить валидацию
        min_overlap: Минимальный overlap слов
        use_vllm: Использовать vLLM вместо HF pipeline (×5-10 throughput на T4)
        vllm_batch_size: Размер батча для vLLM continuous batching
        fast_gpu: Режим для одной L4 (24GB) — больше памяти под vLLM
    """
    # ── Индекс ────────────────────────────────────────────────
    if build_index or not INDEX_PATH.exists():
        logger.info("Building index from scratch")
        websites_df = pd.read_csv(WEBSITES_CSV)
        websites_data = [
            (row["web_id"], row["text"])
            for _, row in websites_df.iterrows()
        ]
        chunks = chunk_all_websites(websites_data)
        logger.info("Created %d chunks", len(chunks))
        indexer = build_and_save_index(chunks)
    else:
        logger.info("Loading existing index")
        indexer = load_index()

    # Очистка памяти после этапа индексации
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleared after indexing")

    # ── Retriever ─────────────────────────────────────────────
    retriever = create_retriever(indexer)

    # Очистка памяти после загрузки reranker'а
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleared after retriever (reranker loaded)")

    # ── Generator ─────────────────────────────────────────────
    hf_model_name = KAGGLE_MODELS.get(llm_model, llm_model)

    # vLLM не умеет читать конфиг fine-tuned модели lirex111/vikhrllama1B_AlfaBank
    # Поэтому для vLLM используем уже слитую плоскую модель.
    if use_vllm and llm_model == "vikhr-1b-finetuned":
        vllm_model_name = "lirex111/vikhrllama1B_AlfaBank_merged"
        logger.info("Using merged model for vLLM: %s", vllm_model_name)
    else:
        vllm_model_name = hf_model_name

    if use_vllm:
        generator = VLLMGenerator(model_name=vllm_model_name, fast_gpu=fast_gpu)
        if fast_gpu:
            logger.info("Using vLLM generator on fastGPU/L4 mode (gpu_memory_utilization=0.80)")
        else:
            logger.info("Using vLLM generator (x5-10 throughput on T4)")
    else:
        generator = KaggleGenerator(model_name=hf_model_name)

    # Финальная очистка памяти перед циклом генерации
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleared after generator init — starting inference loop")

    # ── Кеш ───────────────────────────────────────────────────
    cache = AnswerCache(cache_path)

    # ── Вопросы ───────────────────────────────────────────────
    logger.info("Loading questions from %s", QUESTIONS_CSV)
    questions_df = pd.read_csv(QUESTIONS_CSV)
    total = len(questions_df)

    # ── Цикл генерации ────────────────────────────────────────
    results = []
    stats = {"cached": 0, "generated": 0, "failed": 0, "invalid": 0}
    CHECKPOINT_INTERVAL = 2000

    # Speed guard: stop if generation is too slow to finish in 12h
    SPEED_GUARD_ENABLED = True
    SPEED_GUARD_MIN_GENERATIONS = 20
    SPEED_GUARD_MAX_AVG_SECONDS = 6.5
    generation_start_time: float | None = None
    generation_elapsed = 0.0
    generation_count = 0

    # Проверяем последний чекпоинт
    start_idx = 0
    for cp_num in [6000, 4000, 2000]:
        cp_path = SUBMISSION_CSV.parent / f"submission_checkpoint_{cp_num}.csv"
        if cp_path.exists():
            cp_df = pd.read_csv(cp_path)
            if len(cp_df) == cp_num:
                results = cp_df.to_dict("records")
                start_idx = cp_num
                logger.info("Resuming from checkpoint: %d answers", start_idx)
                break

    # FIX-6: резюм по q_id, а не по индексу (надёжнее)
    done_ids = {str(r["q_id"]) for r in results}

    # ── VLLM batching (continuous batching) ─────────────────
    if use_vllm:
        batch_queries: list[str] = []
        batch_q_ids: list[str] = []
        batch_contexts: list[str] = []

        def process_batch() -> None:
            nonlocal generation_start_time, generation_elapsed, generation_count

            if not batch_queries:
                return

            batch_started = time.perf_counter()
            if generation_start_time is None:
                generation_start_time = batch_started
                logger.info(
                    "Speed guard started | threshold=%.1fs/question | sample_after=%d",
                    SPEED_GUARD_MAX_AVG_SECONDS,
                    SPEED_GUARD_MIN_GENERATIONS,
                )

            try:
                answers = generator.generate_batch(batch_queries, batch_contexts)
                for q_id, query, context, answer in zip(batch_q_ids, batch_queries, batch_contexts, answers):
                    # Шаг 3: Валидация
                    if validate_answers and answer:
                        if not validate_answer(query, answer, min_overlap):
                            fb = extract_answer_from_context(query, context or "")
                            if fb and validate_answer(query, fb, min_overlap):
                                logger.warning(
                                    "Invalid answer for q_id=%s — fallback applied", q_id,
                                )
                                answer = fb
                            else:
                                logger.warning(
                                    "Invalid answer for q_id=%s — no valid fallback, keeping original",
                                    q_id,
                                )
                            stats["invalid"] += 1

                    # Шаг 4: Кешируем
                    if answer:
                        cache.set(query, hf_model_name, q_id, answer)

                    results.append({"q_id": q_id, "answer_new": answer})
                    stats["generated"] += 1
                    generation_count += 1

            except Exception as e:
                logger.error("Batch generation failed: %s", e, exc_info=True)
                # Fallback to per-question processing
                for q_id, query, context in zip(batch_q_ids, batch_queries, batch_contexts):
                    try:
                        answer = generator.generate(query, context)
                    except Exception as inner_e:
                        logger.error("Failed to process q_id=%s: %s", q_id, inner_e, exc_info=True)
                        answer = extract_answer_from_context(query, context or "")
                        if not answer:
                            # Fallback to query-based extraction from context
                            sentences = [s.strip() for s in re.split(r'[.!?»]+', context or "") if s.strip()]
                            answer = sentences[0] if sentences else query
                        stats["failed"] += 1

                    if validate_answers and answer:
                        if not validate_answer(query, answer, min_overlap):
                            fb = extract_answer_from_context(query, context or "")
                            if fb and validate_answer(query, fb, min_overlap):
                                logger.warning(
                                    "Invalid answer for q_id=%s — fallback applied", q_id,
                                )
                                answer = fb
                            else:
                                logger.warning(
                                    "Invalid answer for q_id=%s — no valid fallback, keeping original",
                                    q_id,
                                )
                            stats["invalid"] += 1

                    if answer:
                        cache.set(query, hf_model_name, q_id, answer)

                    results.append({"q_id": q_id, "answer_new": answer})
                    stats["generated"] += 1
                    generation_count += 1

            batch_elapsed = time.perf_counter() - batch_started
            generation_elapsed += batch_elapsed

            if SPEED_GUARD_ENABLED and generation_count >= SPEED_GUARD_MIN_GENERATIONS:
                avg_seconds = generation_elapsed / generation_count
                logger.info(
                    "Speed guard | generated=%d | avg=%.2fs/question | elapsed=%.1fs | projected=%.1fh",
                    generation_count,
                    avg_seconds,
                    generation_elapsed,
                    generation_elapsed * (total - len(results)) / generation_count / 3600,
                )
                if avg_seconds >= SPEED_GUARD_MAX_AVG_SECONDS:
                    raise RuntimeError(
                        f"Generation too slow: avg={avg_seconds:.2f}s/question >= "
                        f"threshold={SPEED_GUARD_MAX_AVG_SECONDS:.2f}s/question. "
                        "Stopping to avoid exceeding 12h session."
                    )

            # Чекпоинт + батч-сохранение кеша
            if (len(results)) % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = SUBMISSION_CSV.parent / f"submission_checkpoint_{len(results)}.csv"
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                cache.save()
                logger.info("Checkpoint saved: %d answers", len(results))

            batch_queries.clear()
            batch_q_ids.clear()
            batch_contexts.clear()

        for _, row in tqdm(questions_df.iterrows(), total=total, desc="Generating"):
            q_id = str(row["q_id"])
            if q_id in done_ids:
                continue
            query = str(row["query"]).strip()

            # Шаг 1: Проверяем кеш
            cached_answer = cache.get(query, hf_model_name)
            if cached_answer is not None:
                results.append({"q_id": q_id, "answer_new": cached_answer})
                stats["cached"] += 1
                continue

            # Шаг 2: Retrieval
            context = None
            try:
                context = retriever.get_context(query)
            except Exception as e:
                logger.error("Retrieval failed for q_id=%s: %s", q_id, e, exc_info=True)
                context = ""
                stats["failed"] += 1

            batch_queries.append(query)
            batch_q_ids.append(q_id)
            batch_contexts.append(context)

            # Flush batch when full
            if len(batch_queries) >= vllm_batch_size:
                process_batch()

        # Flush remaining batch
        process_batch()

    else:
        # ── HF pipeline (old per-question loop) ─────────────────
        if SPEED_GUARD_ENABLED:
            logger.info(
                "Speed guard started | threshold=%.1fs/question | sample_after=%d",
                SPEED_GUARD_MAX_AVG_SECONDS,
                SPEED_GUARD_MIN_GENERATIONS,
            )
        generation_start_time = time.perf_counter()

        for _, row in tqdm(questions_df.iterrows(), total=total, desc="Generating"):
            q_id = str(row["q_id"])
            if q_id in done_ids:
                continue
            query = str(row["query"]).strip()

            # Шаг 1: Проверяем кеш
            cached_answer = cache.get(query, hf_model_name)
            if cached_answer is not None:
                results.append({"q_id": q_id, "answer_new": cached_answer})
                stats["cached"] += 1
                continue

            item_started = time.perf_counter()

            # Шаг 2: Retrieval + Generation
            context = None
            try:
                context = retriever.get_context(query)
                answer = generator.generate(query, context)
            except Exception as e:
                logger.error("Failed to process q_id=%s: %s", q_id, e, exc_info=True)
                answer = extract_answer_from_context(query, context or "")
                if not answer:
                    # Fallback to query-based extraction from context
                    sentences = [s.strip() for s in re.split(r'[.!?»]+', context or "") if s.strip()]
                    answer = sentences[0] if sentences else query
                stats["failed"] += 1

            generation_elapsed += time.perf_counter() - item_started
            generation_count += 1

            if SPEED_GUARD_ENABLED and generation_count >= SPEED_GUARD_MIN_GENERATIONS:
                avg_seconds = generation_elapsed / generation_count
                logger.info(
                    "Speed guard | generated=%d | avg=%.2fs/question | elapsed=%.1fs | projected=%.1fh",
                    generation_count,
                    avg_seconds,
                    generation_elapsed,
                    generation_elapsed * (total - len(results)) / generation_count / 3600,
                )
                if avg_seconds >= SPEED_GUARD_MAX_AVG_SECONDS:
                    raise RuntimeError(
                        f"Generation too slow: avg={avg_seconds:.2f}s/question >= "
                        f"threshold={SPEED_GUARD_MAX_AVG_SECONDS:.2f}s/question. "
                        "Stopping to avoid exceeding 12h session."
                    )

            # Шаг 3: Валидация
            if validate_answers and answer:
                if not validate_answer(query, answer, min_overlap):
                    fb = extract_answer_from_context(query, context or "")
                    if fb and validate_answer(query, fb, min_overlap):
                        logger.warning(
                            "Invalid answer for q_id=%s — fallback applied", q_id,
                        )
                        answer = fb
                    else:
                        logger.warning(
                            "Invalid answer for q_id=%s — no valid fallback, keeping original",
                            q_id,
                        )
                    stats["invalid"] += 1

            # Шаг 4: Кешируем
            if answer:
                cache.set(query, hf_model_name, q_id, answer)

            results.append({"q_id": q_id, "answer_new": answer})
            stats["generated"] += 1

            # Чекпоинт + батч-сохранение кеша
            if (len(results)) % CHECKPOINT_INTERVAL == 0:
                checkpoint_path = SUBMISSION_CSV.parent / f"submission_checkpoint_{len(results)}.csv"
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                cache.save()
                logger.info("Checkpoint saved: %d answers", len(results))

    # ── Итоги ─────────────────────────────────────────────────
    logger.info(
        "Pipeline done | total=%d | cached=%d | generated=%d | failed=%d | invalid=%d",
        total,
        stats["cached"],
        stats["generated"],
        stats["failed"],
        stats["invalid"],
    )

    # ── Сохранение ────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    SUBMISSION_CSV.parent.mkdir(parents=True, exist_ok=True)
    # FIX-5.1: используем колонку answer_new (как в sample_submission.csv)
    results_df.to_csv(SUBMISSION_CSV, index=False)
    logger.info("Results saved to %s (%d rows) with column 'answer_new'", SUBMISSION_CSV, len(results_df))


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle RAG Pipeline")

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build index from scratch",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vikhr-1b-finetuned",
        choices=list(KAGGLE_MODELS.keys()),
        help="LLM model (open-source, Hugging Face)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/answer_cache.json"),
        help="Path to answer cache",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable answer validation",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=1,
        help="Minimum word overlap for validation",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM instead of HF pipeline (×5-10 throughput on T4, install vllm first)",
    )
    parser.add_argument(
        "--vllm-batch-size",
        type=int,
        default=8,
        help="Batch size for vLLM continuous batching (default: 8)",
    )
    parser.add_argument(
        "--fastGPU",
        action="store_true",
        help="Single L4 mode (24GB): increase vLLM gpu_memory_utilization to 0.80",
    )

    args = parser.parse_args()

    run_pipeline(
        build_index=args.build_index,
        llm_model=args.model,
        cache_path=args.cache_path,
        validate_answers=not args.no_validate,
        min_overlap=args.min_overlap,
        use_vllm=args.vllm,
        vllm_batch_size=args.vllm_batch_size,
        fast_gpu=args.fastGPU,
    )