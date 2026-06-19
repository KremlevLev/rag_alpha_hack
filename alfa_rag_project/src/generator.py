"""
Generation module for RAG pipeline.
Handles LLM calls with strict brevity requirements.
"""

import math
import re
from dataclasses import dataclass
from typing import Optional

import razdel
from openai import OpenAI

from config import (
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_SENTENCES,
    MAX_RESPONSE_WORDS,
    MAX_RESPONSE_CHARS,
    TEMPERATURE,
    LLM_TIMEOUT,
)


# ─────────────────────────────────────────────
# Константы и конфигурация
# ─────────────────────────────────────────────

@dataclass
class ExtractorConfig:
    """Настройки извлечения ответа."""
    
    # Минимальная длина информативного предложения (в словах)
    min_sentence_words: int = 2
    
    # Максимальная длина предложения (защита от огромных блоков)
    max_sentence_words: int = 60
    
    # Порог схожести с вопросом (если выше — предложение дублирует вопрос)
    duplicate_threshold: float = 0.7
    
    # Вес позиции предложения в документе (чем раньше — тем важнее)
    position_weight: float = 0.1
    
    # Минимальный score для включения в ответ (очень низкий - всегда что-то возвращаем)
    min_score: float = 0.01
    
    # Количество предложений в итоговом ответе
    max_answer_sentences: int = 2


# Шаблоны "мусорных" предложений
_JUNK_PATTERNS = [
    r"^(а|и|но|да|ну)\s+\w+\s+\w+",          # "А мы вас поддержим"
    r"^(согласно\s+фрагменту|согласно\s+предоставленным\s+фрагментам|в\s+предоставленных\s+фрагментах)\b",
    r"^контекст\s+для\s+ответа\b",
    r"^вопрос\s*:?",
    r"^ответ\s*:?",
    r"^нет\s+ответа\s*(?:\.|[-–—])?\s*(?:контекст|фрагмент)?",
    r"\bнет\s+ответа\b",
    r"узнайте больше",
    r"подробнее на сайте",
    r"свяжитесь с нами",
    r"мы рады помочь",
    r"спасибо за обращение",
    r"^\d+[\.\)]\s*$",                          # Одиночные номера списков
    r"^\s*(?:\d{1,3}\s*[,.)\-]\s*){2,}$",       # Цифровые перечни без содержания
    r"^\s*(?:\d{1,3}\s*){3,}$",                 # Просто набор чисел
    r"^[^\w]+$",                                # Только спецсимволы
]

_JUNK_RE = re.compile(
    "|".join(_JUNK_PATTERNS),
    flags=re.IGNORECASE | re.UNICODE,
)

# Маркеры информативности — предложения с ними ценнее
_INFORMATIVE_MARKERS = [
    r"\b(чтобы|для того чтобы|необходимо|нужно|следует|можно)\b",
    r"\b(зайдите|перейдите|нажмите|выберите|введите|откройте)\b",
    r"\b(позвоните|обратитесь|напишите|заполните)\b",
    r"\b(доступен|доступна|работает|предоставляется)\b",
    r"\b(порядок|способ|шаг|этап|процедура)\b",
]

_INFORMATIVE_RE = re.compile(
    "|".join(_INFORMATIVE_MARKERS),
    flags=re.IGNORECASE | re.UNICODE,
)


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────

def _is_junk_sentence(sentence: str) -> bool:
    """
    Проверяет, является ли предложение мусорным.
    
    Args:
        sentence: Исходное предложение.
        
    Returns:
        True если предложение следует пропустить.
    """
    stripped = sentence.strip()
    
    if not stripped:
        return True
    
    if _JUNK_RE.search(stripped):
        return True
    
    return False


_CONTEXT_HEADER_RE = re.compile(
    r"^\s*Контекст\s+для\s+ответа\s+на\s+вопрос\s*:?\s*",
    flags=re.IGNORECASE | re.UNICODE,
)
_PROMPT_LABEL_RE = re.compile(
    r"^\s*(?:Вопрос|Контекст|Ответ)\s*:?\s*",
    flags=re.IGNORECASE | re.UNICODE,
)
_PASSWORD_RE = re.compile(r"PASSWORD{2,}", flags=re.IGNORECASE)
_TRAILING_CONTEXT_RE = re.compile(r"\s*---\s*Контекст.*$", flags=re.IGNORECASE | re.UNICODE)


def clean_context_sentence(sentence: str) -> str:
    """Remove prompt/context header artifacts from a candidate sentence."""
    sentence = _CONTEXT_HEADER_RE.sub("", sentence).strip()
    sentence = _PASSWORD_RE.sub("", sentence)
    sentence = _TRAILING_CONTEXT_RE.sub("", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def is_prompt_metadata_sentence(sentence: str) -> bool:
    """Return True for prompt leakage lines that are not answer content."""
    return bool(_PROMPT_LABEL_RE.match(sentence.strip()))


def _is_duplicate_of_query(
    sentence_normalized: str,
    query_words: set[str],
    threshold: float,
) -> bool:
    """
    Проверяет, является ли предложение перефразировкой вопроса.
    
    Логика: если >70% слов вопроса присутствует в предложении
    И предложение короче вопроса × 1.5 — скорее всего это повтор.
    
    Args:
        sentence_normalized: Нормализованное предложение.
        query_words: Множество слов вопроса.
        threshold: Порог совпадения.
        
    Returns:
        True если предложение дублирует вопрос.
    """
    if not query_words:
        return False
    
    sentence_words = set(sentence_normalized.split())
    
    # Защита: очень длинные предложения не могут быть дублями вопроса
    if len(sentence_words) > len(query_words) * 2:
        return False
    
    overlap = sum(
        1 for word in query_words
        if word_matches(word, sentence_normalized)
    )
    similarity = overlap / len(query_words)
    
    return similarity >= threshold


def _precompute_doc_freq(
    query_words: set[str],
    all_sentences_normalized: list[str],
) -> dict[str, float]:
    """
    Предрасчитывает IDF для каждого слова запроса (один проход по всем предложениям).
    
    Args:
        query_words: Слова запроса.
        all_sentences_normalized: Все нормализованные предложения.
        
    Returns:
        Словарь {слово: idf_value}.
    """
    total = len(all_sentences_normalized)
    if total == 0 or not query_words:
        return {}
    
    doc_freq: dict[str, int] = {w: 0 for w in query_words}
    for s in all_sentences_normalized:
        for word in query_words:
            if word_matches(word, s):
                doc_freq[word] += 1
    
    return {
        word: math.log((total + 1) / (freq + 1)) + 1.0
        for word, freq in doc_freq.items()
    }


def _compute_tfidf_score(
    query_words: set[str],
    sentence_normalized: str,
    precomputed_idf: dict[str, float],
) -> float:
    """
    Вычисляет TF-IDF подобный score релевантности предложения.
    Использует предрасчитанные IDF — не пересчитывает на каждое предложение.
    
    Args:
        query_words: Слова запроса.
        sentence_normalized: Текущее предложение (нормализованное).
        precomputed_idf: Словарь {слово: idf} из _precompute_doc_freq.
        
    Returns:
        Score от 0.0 до 1.0+.
    """
    if not query_words or not sentence_normalized:
        return 0.0
    
    score = 0.0
    for word in query_words:
        tf = 1.0 if word_matches(word, sentence_normalized) else 0.0
        if tf == 0.0:
            continue
        idf = precomputed_idf.get(word, 0.0)
        score += tf * idf
    
    return score / len(query_words) if query_words else 0.0


def _informative_bonus(sentence: str) -> float:
    """
    Возвращает бонус за информативность предложения.
    
    Args:
        sentence: Исходное предложение.
        
    Returns:
        Бонус от 0.0 до 0.3.
    """
    matches = _INFORMATIVE_RE.findall(sentence)
    # Каждый маркер даёт +0.1, максимум 0.3
    return min(len(matches) * 0.1, 0.3)


# ─────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────

@dataclass
class _ScoredSentence:
    """Предложение с его score для сортировки."""
    text: str
    score: float
    position: int  # Позиция в оригинальном тексте


# System prompt для русскоязычных моделей
SYSTEM_PROMPT = """Ты — банковский AI-ассистент Альфа-Банка.

Отвечай только на вопрос клиента. Используй контекст, но не копируй его дословно.
Если точного ответа нет, не пиши «Нет ответа» и не копируй служебные фрагменты:
дай короткий полезный ответ по смыслу или вежливо попроси уточнить детали вопроса.

Формат ответа: только сам ответ. Не возвращай служебные строки вида "Вопрос:", "Контекст:", "Ответ:", "Ответь кратко" или "на основе контекста".
Не вставляй цифровые перечни без содержания, мусорные повторы или технические фрагменты.
""".strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison (ё -> е).
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return text.lower().replace('ё', 'е')


def truncate_to_words(text: str, max_words: int) -> str:
    """
    Truncate text to maximum word count.
    
    Args:
        text: Input text
        max_words: Maximum number of words
        
    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Keep complete words, try to end at sentence boundary
    truncated = " ".join(words[:max_words])
    
    # Try to find last sentence end
    for punct in [".", "!", "?"]:
        last_punct = truncated.rfind(punct)
        if last_punct > len(truncated) * 0.5:  # At least half the text
            return truncated[:last_punct + 1]
    
    return truncated


def truncate_to_chars(text: str, max_chars: int) -> str:
    """
    Truncate text to maximum character count.
    
    This is a SAFETY limit to prevent 3x length penalty.
    Cuts at the last complete sentence boundary before the limit.
    
    Args:
        text: Input text
        max_chars: Maximum number of characters
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    
    # Try to find last sentence end before the limit
    truncated = text[:max_chars]
    
    # Find the last sentence-ending punctuation
    for punct in [".", "!", "?", "»"]:
        last_punct = truncated.rfind(punct)
        if last_punct > max_chars * 0.3:  # At least 30% of the limit
            return truncated[:last_punct + 1]
    
    # If no good sentence boundary, find last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.5:  # At least half the limit
        return truncated[:last_space]
    
    # Hard cut as last resort
    return truncated


def truncate_to_sentences(text: str, max_sentences: int) -> str:
    """
    Truncate text to maximum number of sentences.
    
    This is the PRIMARY truncation for BERT-Recall-L compliance.
    Preserves complete sentences for maximum information density.
    
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


def word_matches(word: str, text: str) -> bool:
    """
    Check if word (or its root) is in text.
    Handles Russian morphology: счета -> счет, карты -> карта.
    
    Args:
        word: Word to search for
        text: Text to search in
        
    Returns:
        True if word found
    """
    # Direct match
    if word in text:
        return True
    
    # Try removing common Russian endings
    for ending in ['а', 'ы', 'и', 'ов', 'ев', 'ой', 'ий', 'ый', 'ом', 'ем', 'ую', 'юю']:
        if word.endswith(ending) and word[:-len(ending)] in text:
            return True
    
    return False


def clean_sentence(sentence: str) -> str:
    """
    Remove chunk ID prefix from sentence.
    
    Args:
        sentence: Sentence with optional [chunk_id] prefix
        
    Returns:
        Cleaned sentence
    """
    # Remove [number] prefix
    if sentence.startswith('['):
        end_bracket = sentence.find('] ')
        if end_bracket > 0:
            return sentence[end_bracket + 2:]
    return sentence


def extract_answer_from_context(
    query: str,
    context: str,
    config: Optional[ExtractorConfig] = None,
) -> str:
    """
    Извлекает релевантный ответ из контекста по запросу.
    
    Алгоритм:
        1. Разбивает контекст на предложения через razdel.
        2. Фильтрует мусорные и слишком короткие предложения.
        3. Исключает предложения, дублирующие вопрос.
        4. Ранжирует по TF-IDF + позиция + информативность.
        5. Собирает топ-N предложений в связный ответ.
        6. ПРИМЕНЯЕТ обрезку по предложениям.
    
    Args:
        query: Вопрос пользователя.
        context: Текст документа/FAQ.
        config: Настройки извлечения (опционально).
        
    Returns:
        Строка с ответом (всегда что-то возвращает, даже если score низкий).
        
    Examples:
        >>> extract_answer_from_context(
        ...     "Как узнать номер счёта?",
        ...     "Номер счёта доступен в личном кабинете. Зайдите в раздел 'Счета'.",
        ... )
        "Номер счёта доступен в личном кабинете."
    """
    if config is None:
        config = ExtractorConfig()
    
    # ── Шаг 1: Подготовка данных ──────────────────────────────
    
    query_normalized = normalize_text(query)
    query_words = set(query_normalized.split())
    
    # Убираем стоп-слова из вопроса для более точного matching
    # (короткие слова типа "как", "что", "где" — малоинформативны)
    meaningful_query_words = {w for w in query_words if len(w) > 2}
    
    if not meaningful_query_words:
        # Вопрос состоит только из стоп-слов — работаем с тем что есть
        meaningful_query_words = query_words
    
    # ── Шаг 2: Разбивка на предложения ───────────────────────
    
    raw_sentences = [
        clean_context_sentence(s.text.strip())
        for s in razdel.sentenize(context)
        if s.text.strip()
    ]
    raw_sentences = [
        sentence for sentence in raw_sentences
        if sentence and not is_prompt_metadata_sentence(sentence)
    ]
    
    if not raw_sentences:
        return ""
    
    # ── Шаг 3: Предрасчёт IDF (один проход, а не O(N²)) ──────
    
    normalized_sentences = [normalize_text(s) for s in raw_sentences]
    precomputed_idf = _precompute_doc_freq(meaningful_query_words, normalized_sentences)
    
    # ── Шаг 4: Scoring ВСЕХ предложений (без строгой фильтрации) ──
    
    scored: list[_ScoredSentence] = []
    total_sentences = len(raw_sentences)
    
    for idx, sentence in enumerate(raw_sentences):
        words_count = len(sentence.split())
        
        # Пропускаем только явно мусорные
        if _is_junk_sentence(sentence):
            continue
        
        # Пропускаем очень длинные и короткие
        if words_count < 1 or words_count > config.max_sentence_words:
            continue
        
        sentence_normalized = normalized_sentences[idx]
        
        # TF-IDF релевантность (используем предрасчитанный IDF)
        tfidf_score = _compute_tfidf_score(
            meaningful_query_words,
            sentence_normalized,
            precomputed_idf,
        )
        
        # Штраф за позицию (предложения в начале документа важнее)
        position_factor = 1.0 - (idx / total_sentences) * config.position_weight
        
        # Бонус за информативность
        informative_bonus = _informative_bonus(sentence)
        
        total_score = (tfidf_score * position_factor) + informative_bonus
        
        scored.append(_ScoredSentence(
            text=sentence,
            score=total_score,
            position=idx,
        ))
    
    if not scored:
        # Все предложения мусор - возвращаем первое приличное
        for sentence in raw_sentences:
            cleaned = clean_context_sentence(sentence)
            if cleaned and len(cleaned.split()) >= 2 and not _is_junk_sentence(cleaned) and not is_prompt_metadata_sentence(cleaned):
                return truncate_to_sentences(cleaned, MAX_SENTENCES)
        return ""
    
    # ── Шаг 4: Выбор топ-N и восстановление порядка ───────────
    
    # Сортируем по score (лучшие первые)
    top_sentences = sorted(scored, key=lambda s: s.score, reverse=True)
    top_sentences = top_sentences[:config.max_answer_sentences]
    
    # Восстанавливаем оригинальный порядок для связности текста
    top_sentences.sort(key=lambda s: s.position)
    
    # ── Шаг 5: Сборка ответа ──────────────────────────────────
    
    answer = " ".join(s.text for s in top_sentences)
    answer = clean_context_sentence(answer)
    answer = _PROMPT_LABEL_RE.sub("", answer).strip()
    if is_prompt_metadata_sentence(answer):
        return ""
    
    # ── Шаг 6: Обрезка по предложениям (основной лимит) ───────
    
    answer = truncate_to_sentences(answer, MAX_SENTENCES)
    
    # ── Шаг 7: Safety обрезка по символам ─────────────────────
    
    answer = truncate_to_chars(answer, MAX_RESPONSE_CHARS)
    
    return answer


class Generator:
    """
    Handles LLM generation with brevity constraints.
    """
    
    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
    ):
        """
        Initialize generator with LLM client.
        
        Args:
            base_url: OpenAI-compatible API base URL
            model: Model name
        """
        self.client = OpenAI(base_url=base_url, api_key="ollama", timeout=LLM_TIMEOUT)
        self.model = model
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate answer for query using context.
        
        Args:
            query: User question
            context: Retrieved context from retriever
            
        Returns:
            Generated answer (truncated to max sentences and chars)
        """
        if not context:
            return extract_answer_from_context(query, context)
        
        user_prompt = f"{query}\n\n{context}".strip()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=256,  # Conservative limit
                timeout=LLM_TIMEOUT,
            )
            
            answer = response.choices[0].message.content or ""
            
            # Post-process: truncate to max sentences first, then chars
            answer = truncate_to_sentences(answer, MAX_SENTENCES)
            answer = truncate_to_chars(answer, MAX_RESPONSE_CHARS)
            
            return answer.strip()
            
        except Exception as e:
            # Fallback: extract from context (ALWAYS returns something)
            return extract_answer_from_context(query, context)


def create_generator(
    base_url: str = LLM_BASE_URL,
    model: str = LLM_MODEL,
) -> Generator:
    """
    Convenience function to create generator.
    
    Args:
        base_url: OpenAI-compatible API base URL
        model: Model name
        
    Returns:
        Generator instance
    """
    return Generator(base_url, model)