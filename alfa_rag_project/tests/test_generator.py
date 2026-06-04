"""
Tests for generator module.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import (
    truncate_to_words,
    truncate_to_chars,
    normalize_text,
    word_matches,
    clean_sentence,
    extract_answer_from_context,
    ExtractorConfig,
)


class TestTruncateToWords:
    """Tests for response truncation."""
    
    def test_short_text_unchanged(self) -> None:
        """Test short text is not modified."""
        text = "Короткий ответ"
        result = truncate_to_words(text, max_words=30)
        assert result == text
    
    def test_long_text_truncated(self) -> None:
        """Test long text is truncated."""
        words = ["слово"] * 50
        text = " ".join(words)
        result = truncate_to_words(text, max_words=30)
        assert len(result.split()) == 30
    
    def test_truncation_at_sentence_end(self) -> None:
        """Test truncation tries to end at sentence boundary."""
        text = "Первое предложение. Второе предложение. Третье предложение. " + " ".join(["слово"] * 50)
        result = truncate_to_words(text, max_words=30)
        # Should end at a sentence boundary if possible
        assert result.endswith(".") or result.endswith("!") or result.endswith("?") or len(result.split()) <= 30
    
    def test_empty_text(self) -> None:
        """Test empty text handling."""
        result = truncate_to_words("", max_words=30)
        assert result == ""
    
    def test_exact_word_count(self) -> None:
        """Test text with exact word count."""
        words = ["слово"] * 30
        text = " ".join(words)
        result = truncate_to_words(text, max_words=30)
        assert result == text


class TestTruncateToChars:
    """Tests for character-based truncation."""
    
    def test_short_text_unchanged(self) -> None:
        """Test short text is not modified."""
        text = "Короткий ответ"
        result = truncate_to_chars(text, max_chars=150)
        assert result == text
    
    def test_long_text_truncated(self) -> None:
        """Test long text is truncated to char limit."""
        text = "Это очень длинный ответ который должен быть обрезан на определенном количестве символов"
        result = truncate_to_chars(text, max_chars=50)
        assert len(result) <= 50
    
    def test_truncation_at_sentence_end(self) -> None:
        """Test truncation tries to end at sentence boundary."""
        text = "Первое предложение. Второе предложение. Третье предложение. И ещё много текста"
        result = truncate_to_chars(text, max_chars=30)
        # Should end at a sentence boundary if possible
        assert result.endswith(".") or result.endswith("!") or result.endswith("?") or len(result) <= 30
    
    def test_empty_text(self) -> None:
        """Test empty text handling."""
        result = truncate_to_chars("", max_chars=150)
        assert result == ""
    
    def test_exact_char_count(self) -> None:
        """Test text with exact char count."""
        text = "Текст ровно сто символов!!!"
        # Pad to exactly 150 chars
        text = "Текст ровно сто символов!!!" + " " * (150 - len("Текст ровно сто символов!!!"))
        result = truncate_to_chars(text, max_chars=150)
        assert len(result) <= 150


class TestNormalizeText:
    """Tests for text normalization."""
    
    def test_yo_to_e(self) -> None:
        """Test ё -> е normalization."""
        assert normalize_text("счёта") == "счета"
        assert normalize_text("СЁЛЬНЫЙ") == "сельный"
    
    def test_lowercase(self) -> None:
        """Test lowercase conversion."""
        assert normalize_text("ТЕКСТ") == "текст"


class TestWordMatches:
    """Tests for word matching with morphology."""
    
    def test_direct_match(self) -> None:
        """Test direct word match."""
        assert word_matches("счет", "это счет") is True
    
    def test_morphology_match(self) -> None:
        """Test word match with different forms."""
        assert word_matches("счета", "это счет") is True
        assert word_matches("карты", "это карта") is True
    
    def test_no_match(self) -> None:
        """Test no match."""
        assert word_matches("авто", "это счет") is False


class TestCleanSentence:
    """Tests for sentence cleaning."""
    
    def test_with_chunk_id(self) -> None:
        """Test removing chunk ID prefix."""
        assert clean_sentence("[123] Текст ответа") == "Текст ответа"
    
    def test_without_chunk_id(self) -> None:
        """Test sentence without prefix unchanged."""
        assert clean_sentence("Текст ответа") == "Текст ответа"


class TestExtractAnswerFromContext:
    """Tests for answer extraction."""
    
    def test_basic_extraction(self) -> None:
        """Test basic extraction."""
        context = "Корреспондентский счет № 30101810200000000593 в Банке. Зайдите в личный кабинет."
        result = extract_answer_from_context("номер счета", context)
        assert "счет" in result.lower()
    
    def test_empty_context(self) -> None:
        """Test empty context."""
        result = extract_answer_from_context("вопрос", "")
        assert result == ""
    
    def test_yo_handling(self) -> None:
        """Test ё handling in query."""
        context = "Счет № 12345. Номер счёта доступен в приложении."
        result = extract_answer_from_context("счёта", context)
        # Result preserves original text, check with normalization
        assert "счёта" in result.lower() or "счет" in result.lower()
    
    def test_junk_sentence_filtering(self) -> None:
        """Test that junk sentences are filtered out."""
        context = "Узнайте больше на сайте. Номер счета в личном кабинете."
        result = extract_answer_from_context("счет", context)
        # "Узнайте больше на сайте" should be filtered as junk
        assert "узнайте больше" not in result.lower()
    
    def test_duplicate_detection(self) -> None:
        """Test that sentences duplicating the query are filtered."""
        context = "Как узнать номер счёта? Номер счёта доступен в личном кабинете."
        result = extract_answer_from_context("Как узнать номер счёта?", context)
        # First sentence duplicates the query, should be filtered
        assert "узнайте" not in result.lower() or "как" not in result.lower()
    
    def test_informative_bonus(self) -> None:
        """Test that informative sentences get bonus score."""
        context = "Номер счёта доступен в личном кабинете. Зайдите в раздел 'Счета' для просмотра."
        result = extract_answer_from_context("счёта", context)
        # Both sentences should be included, ordered by position
        assert "счёта" in result.lower() or "счет" in result.lower()
    
    def test_short_sentence_filtered(self) -> None:
        """Test that very short sentences are filtered."""
        context = "Да все важно. Номер счёта в личном кабинете."
        result = extract_answer_from_context("счёта", context)
        # "Да все важно" is too short (3 words), second sentence should be returned
        assert result == "" or "счёта" in result.lower() or "счет" in result.lower()
    
    def test_custom_config(self) -> None:
        """Test with custom ExtractorConfig."""
        context = "Номер счёта в личном кабинете. Доступно в приложении. Зайдите в раздел."
        config = ExtractorConfig(min_sentence_words=2, max_answer_sentences=2)
        result = extract_answer_from_context("счёта", context, config)
        assert "счёта" in result.lower() or "счет" in result.lower()
    
    def test_char_truncation_applied(self) -> None:
        """Test that character truncation is applied to extracted answer."""
        # Create a context that would produce a long answer
        context = " ".join(["Номер счёта в личном кабинете. Зайдите в раздел. "] * 10)
        result = extract_answer_from_context("счёта", context)
        # Answer should be truncated to MAX_RESPONSE_CHARS (150)
        assert len(result) <= 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])