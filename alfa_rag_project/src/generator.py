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
    r"узнайте больше",
    r"подробнее на сайте",
    r"свяжитесь с нами",
    r"мы рады помочь",
    r"спасибо за обращение",
    r"^\d+[\.\)]\s*$",                          # Одиночные номера списков
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
    
    # Проверка по паттернам
    if _JUNK_RE.search(stripped):
        return True
    
    return False


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


def _compute_tfidf_score(
    query_words: set[str],
    sentence_normalized: str,
    all_sentences_normalized: list[str],
) -> float:
    """
    Вычисляет TF-IDF подобный score релевантности предложения.
    
    Вместо чистого подсчёта совпадений учитывает редкость слова
    в документе — частые слова ("банк", "счёт") весят меньше.
    
    Args:
        query_words: Слова запроса.
        sentence_normalized: Текущее предложение (нормализованное).
        all_sentences_normalized: Все предложения для подсчёта IDF.
        
    Returns:
        Score от 0.0 до 1.0+.
    """
    if not query_words or not sentence_normalized:
        return 0.0
    
    total_sentences = len(all_sentences_normalized)
    score = 0.0
    
    for word in query_words:
        # TF: слово присутствует в предложении?
        tf = 1.0 if word_matches(word, sentence_normalized) else 0.0
        
        if tf == 0.0:
            continue
        
        # IDF: насколько слово редкое в документе?
        doc_freq = sum(
            1 for s in all_sentences_normalized
            if word_matches(word, s)
        )
        # +1 сглаживание, чтобы избежать деления на ноль
        idf = math.log((total_sentences + 1) / (doc_freq + 1)) + 1.0
        
        score += tf * idf
    
    # Нормализация по количеству слов запроса
    return score / len(query_words)


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


# System prompt enforcing strict context-based answers with few-shot examples
SYSTEM_PROMPT = """
Ты суровый банковский AI-аналитик. Отвечай на вопрос строго на основе предоставленного текста.

ПРАВИЛА:
1. Выдавай ТОЛЬКО факты из контекста. Никаких приветствий и лишних слов.
2. Если в контексте нет прямого ответа, выбери самое релевантное предложение из контекста.
3. Отвечай максимально емко. Объединяй длинные списки в одно-два предложения через запятую.
4. Твой ответ должен быть ОДНО-ДВУХ предложЕНИЯМИ.

ПРИМЕРЫ:
Вопрос: Как узнать номер счёта?
Контекст: Номер счёта отображается в мобильном приложении банка на вкладке "Мои счета". Также вы можете позвонить в поддержку.
Ответ: Номер счёта отображается в мобильном приложении банка на вкладке "Мои счета".

Вопрос: Что такое БИК?
Контекст: БИК — это банковский идентификационный код, используемый для перечисления средств.
Ответ: БИК — это банковский идентификационный код, используемый для перечисления средств.

Вопрос: Как открыть вклад?
Контекст: Для открытия вклада необходимо подъехать в отделение банка с паспортом и заполнить форму.
Ответ: Для открытия вклада необходимо подъехать в отделение банка с паспортом и заполнить форму.
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
        s.text.strip()
        for s in razdel.sentenize(context)
        if s.text.strip()
    ]
    
    if not raw_sentences:
        return ""
    
    # ── Ѐаг 3: Scoring ВСЕХ предложений (без строгой фильтрации) ──
    
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
        
        sentence_normalized = normalize_text(sentence)
        
        # TF-IDF релевантность
        tfidf_score = _compute_tfidf_score(
            meaningful_query_words,
            sentence_normalized,
            [normalize_text(s) for s in raw_sentences],
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
            if len(sentence.split()) >= 2 and not _is_junk_sentence(sentence):
                return truncate_to_sentences(sentence, MAX_SENTENCES)
        return truncate_to_sentences(raw_sentences[0], MAX_SENTENCES) if raw_sentences else ""
    
    # ── Шаг 4: Выбор топ-N и восстановление порядка ───────────
    
    # Сортируем по score (лучшие первые)
    top_sentences = sorted(scored, key=lambda s: s.score, reverse=True)
    top_sentences = top_sentences[:config.max_answer_sentences]
    
    # Восстанавливаем оригинальный порядок для связности текста
    top_sentences.sort(key=lambda s: s.position)
    
    # ── Шаг 5: Сборка ответа ──────────────────────────────────
    
    answer = " ".join(s.text for s in top_sentences)
    
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
        
        user_prompt = f"""
Вопрос: {query}

Контекст:
{context}

Ответь кратко на основе контекста.
""".strip()
        
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