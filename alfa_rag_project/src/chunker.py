"""
Text chunking module for RAG pipeline.
Splits cleaned documents into overlapping chunks for indexing.
"""

import re
import html
import logging
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import razdel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────

@dataclass
class ChunkerConfig:
    """
    Настройки чанкера.

    Attributes:
        chunk_size: Целевой размер чанка в символах.
            450 — баланс между полнотой и точностью retrieval.
            При 450 чанк содержит 3-5 предложений, из которых
            релевантны обычно 1-2. Остальные — шум для эмбеддинга.
        chunk_overlap: Перекрытие между соседними чанками.
            100 символов — достаточно для сохранения контекста на границе.
        min_chunk_length: Минимальная длина чанка после очистки.
            Чанки короче этого значения отбрасываются.
        keep_sentence_boundary: Не разрывать предложения.
            Чанк всегда заканчивается на границе предложения.
    """
    chunk_size: int = 650
    chunk_overlap: int = 120
    min_chunk_length: int = 40
    keep_sentence_boundary: bool = True


# ─────────────────────────────────────────────
# Паттерны для clean_text()
# ─────────────────────────────────────────────

# HTML-теги
_HTML_TAG_RE = re.compile(r"<[^>]{1,100}>", re.UNICODE)

# Спецпробелы: \xa0, zero-width, табы
_WHITESPACE_VARIANTS_RE = re.compile(
    r"[\xa0\u00a0\u200b\u200c\u200d\u2060\ufeff\t]+",
    re.UNICODE,
)

# Повторяющиеся пробелы
_MULTI_SPACE_RE = re.compile(r" {2,}", re.UNICODE)

# Повторяющиеся переносы строк
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}", re.UNICODE)

# Служебные фразы банковских FAQ которые засоряют чанки.
# Важно: фразы упорядочены от длинных к коротким
# чтобы более специфичные паттерны матчились первыми.
_SERVICE_PHRASES = [
    # Уведомления и предупреждения
    r"потребуется уточнить\s+[а-яёa-z\s,]+?(?=[.!?\n]|$)",
    r"обратите внимание[,:]?\s*",
    r"обращаем ваше внимание[,:]?\s*",
    r"уточните у специалиста[.!]?\s*",
    r"при необходимости уточните[^.!?\n]*[.!?]?\s*",
    
    # Рекламные и призывные
    r"узнайте больше\s+(?:на сайте|в приложении|по телефону)[^.!?\n]*[.!?]?\s*",
    r"подробнее\s+(?:на сайте|в разделе|по ссылке)[^.!?\n]*[.!?]?\s*",
    r"(?:мы\s+)?рады\s+(?:вам\s+)?помочь[.!]?\s*",
    r"свяжитесь с нами[^.!?\n]*[.!?]?\s*",
    r"наши специалисты\s+(?:готовы|помогут)[^.!?\n]*[.!?]?\s*",
    
    # Технические артефакты
    r"\[\s*\d+\s*\]",                # chunk ID: [38644]
    r"<!-{2}.*?-{2}>",               # HTML-комментарии
]

_SERVICE_PHRASE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _SERVICE_PHRASES),
    flags=re.IGNORECASE | re.UNICODE | re.DOTALL,
)


# ─────────────────────────────────────────────
# Функции очистки
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Очищает текст перед чанкингом.

    Pipeline:
        1. HTML unescape → декодируем &nbsp; & и т.д.
        2. Убираем HTML-теги
        3. Убираем служебные фразы
        4. Нормализуем пробельные символы
        5. Финальный strip

    Args:
        text: Сырой текст документа.

    Returns:
        Очищенный текст, готовый к чанкингу.

    Examples:
        >>> clean_text("<p>Счёт&nbsp;открыт.</p> Обратите внимание: БИК.")
        "Счёт открыт."
    """
    if not text:
        return ""

    # 1. HTML entities: &nbsp; → пробел, & → &
    text = html.unescape(text)

    # 2. HTML теги → пробел (не пустота, чтобы не склеить слова)
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Служебные фразы
    text = _SERVICE_PHRASE_RE.sub(" ", text)

    # 4. Спецпробелы → обычный пробел
    text = _WHITESPACE_VARIANTS_RE.sub(" ", text)

    # 5. Схлопываем повторяющиеся пробелы
    text = _MULTI_SPACE_RE.sub(" ", text)

    # 6. Схлопываем повторяющиеся переносы строк
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


# ─────────────────────────────────────────────
# Чанкер
# ─────────────────────────────────────────────

class Chunker:
    """
    Разбивает очищенный текст на перекрывающиеся чанки.

    Стратегия:
        Накапливаем предложения пока не достигнем chunk_size.
        Сохраняем последние overlap символов как начало следующего чанка.
        Чанки всегда начинаются и заканчиваются на границе предложения.
    """

    def __init__(self, config: ChunkerConfig | None = None):
        """
        Args:
            config: Настройки чанкера. По умолчанию — ChunkerConfig().
        """
        self.config = config or ChunkerConfig()

        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.config.chunk_overlap}) must be "
                f"less than chunk_size ({self.config.chunk_size})"
            )

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Разбивает текст на предложения через razdel.

        Фильтрует пустые и слишком короткие фрагменты.

        Args:
            text: Очищенный текст.

        Returns:
            Список предложений.
        """
        sentences = []
        for sent in razdel.sentenize(text):
            sentence = sent.text.strip()
            # Пропускаем однословные фрагменты — это артефакты разбивки
            if sentence and len(sentence.split()) > 1:
                sentences.append(sentence)
        return sentences

    def _build_chunks_from_sentences(
        self,
        sentences: list[str],
    ) -> Iterator[str]:
        """
        Собирает чанки из предложений с перекрытием.

        Алгоритм:
            1. Накапливаем предложения в буфер.
            2. Когда буфер достигает chunk_size — эмитируем чанк.
            3. Откатываемся на overlap символов (целыми предложениями).
            4. Повторяем.

        Args:
            sentences: Список предложений.

        Yields:
            Текст чанка.
        """
        if not sentences:
            return

        buffer: list[str] = []
        buffer_length: int = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Предложение само по себе длиннее chunk_size
            # Эмитируем текущий буфер и само предложение отдельно
            if sentence_length > self.config.chunk_size:
                if buffer:
                    yield " ".join(buffer)
                    buffer, buffer_length = [], 0

                logger.debug(
                    "Long sentence (%d chars) emitted as standalone chunk",
                    sentence_length,
                )
                yield sentence
                continue

            # Добавляем предложение в буфер
            # +1 для пробела между предложениями
            projected_length = buffer_length + sentence_length + (1 if buffer else 0)

            if projected_length > self.config.chunk_size and buffer:
                # Буфер заполнен — эмитируем чанк
                yield " ".join(buffer)

                # Откат: убираем предложения с начала буфера
                # пока хвост буфера >= overlap
                while buffer and buffer_length > self.config.chunk_overlap:
                    removed = buffer.pop(0)
                    buffer_length -= len(removed) + 1  # +1 за пробел

                # Добавляем текущее предложение к "хвосту"
                buffer.append(sentence)
                buffer_length += sentence_length + (1 if len(buffer) > 1 else 0)
            else:
                buffer.append(sentence)
                buffer_length = projected_length

        # Остаток в буфере
        if buffer:
            yield " ".join(buffer)

    def chunk_text(self, text: str) -> list[str]:
        """
        Полный pipeline: очистка → разбивка → фильтрация.

        Args:
            text: Сырой текст документа.

        Returns:
            Список очищенных чанков, готовых к индексированию.

        Examples:
            >>> chunker = Chunker()
            >>> chunker.chunk_text("<p>Счёт открыт.</p> Обратите внимание: БИК.")
            ["Счёт открыт."]
        """
        # Шаг 1: очистка
        cleaned = clean_text(text)

        if not cleaned:
            logger.debug("Text is empty after cleaning")
            return []

        # Шаг 2: разбивка на предложения
        sentences = self._split_into_sentences(cleaned)

        if not sentences:
            logger.debug("No sentences found after splitting")
            return []

        # Шаг 3: сборка чанков
        raw_chunks = self._build_chunks_from_sentences(sentences)

        # Шаг 4: фильтрация слишком коротких чанков
        chunks = [
            chunk for chunk in raw_chunks
            if len(chunk) >= self.config.min_chunk_length
        ]

        logger.debug(
            "Produced %d chunks from %d sentences",
            len(chunks),
            len(sentences),
        )

        return chunks


# ─────────────────────────────────────────────
# Legacy API (backward compatibility)
# ─────────────────────────────────────────────

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    chunk_id: int
    web_id: int
    text: str
    parent_id: int | None = None
    parent_text: str | None = None
    parent_start: int | None = None
    parent_end: int | None = None


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using razdel.sentenize.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences without empty strings
    """
    sentences = razdel.sentenize(text)
    return [sent.text.strip() for sent in sentences if sent.text.strip()]


def create_chunks(web_id: int, text: str, start_chunk_id: int = 0) -> List[Chunk]:
    """
    Create chunks from text without breaking sentences.
    
    Uses sliding window approach with overlap, ensuring sentences
    are not split across chunks.
    
    Args:
        web_id: Source document ID
        text: Input text to chunk
        start_chunk_id: Starting ID for chunks (for batch processing)
        
    Returns:
        List of Chunk objects with sequential IDs
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks: List[Chunk] = []
    current_chunk_id = start_chunk_id
    current_position = 0
    
    while current_position < len(sentences):
        # Build chunk by adding sentences until we reach CHUNK_SIZE
        chunk_text = ""
        chunk_sentences: List[str] = []
        
        for i in range(current_position, len(sentences)):
            sentence = sentences[i]
            
            # Check if adding this sentence would exceed chunk size
            # (accounting for space separator if not first sentence)
            potential_text = chunk_text + (" " if chunk_text else "") + sentence
            
            if len(potential_text) > CHUNK_SIZE and chunk_text:
                # Stop here - we have enough content
                break
            
            chunk_text = potential_text
            chunk_sentences.append(sentence)
        
        # Create chunk if we have content
        if chunk_text:
            chunks.append(Chunk(
                chunk_id=current_chunk_id,
                web_id=web_id,
                text=chunk_text
            ))
            current_chunk_id += 1
        
        # Move position forward, accounting for overlap
        # Overlap means we go back N sentences to create context
        if current_position + len(chunk_sentences) >= len(sentences):
            break
        
        # Calculate overlap: go back by approximately CHUNK_OVERLAP characters worth of sentences
        overlap_chars = 0
        overlap_sentences = 0
        
        for i in range(len(chunk_sentences) - 1, -1, -1):
            overlap_chars += len(chunk_sentences[i])
            overlap_sentences += 1
            if overlap_chars >= CHUNK_OVERLAP:
                break
        
        # Move forward, but keep some overlap
        current_position = current_position + len(chunk_sentences) - min(overlap_sentences, len(chunk_sentences) - 1)
    
    return chunks


def chunk_website(web_id: int, text: str) -> List[Chunk]:
    """
    Convenience function to chunk a single website.
    
    Args:
        web_id: Website ID
        text: Website text content
        
    Returns:
        List of chunks for this website
    """
    return create_chunks(web_id, text)


def chunk_all_websites(websites_data: List[Tuple[int, str]]) -> List[Chunk]:
    """
    Chunk all websites using the OЧИЩАЮЩИЙ Chunker (not legacy create_chunks).

    Each chunk goes through clean_text() pipeline: HTML unescape → tag removal →
    service phrase removal → whitespace normalization → sentence splitting.

    Args:
        websites_data: List of (web_id, text) tuples.

    Returns:
        List of clean Chunk objects with unique sequential IDs.
    """
    chunker = Chunker(ChunkerConfig(
        chunk_size=CHUNK_SIZE,  # FIX-C2: цельный FAQ-ответ влезает в один чанк
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_length=40,
    ))
    all_chunks: List[Chunk] = []
    next_id = 0

    for web_id, text in websites_data:
        for piece in chunker.chunk_text(text):  # clean_text() вызывается внутри
            all_chunks.append(Chunk(chunk_id=next_id, web_id=web_id, text=piece))
            next_id += 1

    logger.info(
        "Chunked %d websites into %d chunks (Chunker with clean_text)",
        len(websites_data),
        len(all_chunks),
    )
    return all_chunks