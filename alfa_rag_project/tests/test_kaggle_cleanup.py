"""
Tests for Kaggle output cleanup and reference rescue fallback.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kaggle_main import clean_reference_answer, finalize_generated_answer, is_garbage_answer


class TestGarbageDetection:
    """Tests for corrupted LLM output detection."""

    def test_detects_latin_cyrillic_corruption(self) -> None:
        answer = "Номер undergoingётного счёта можно узнать в личном кабинете"
        assert is_garbage_answer(answer) is True

    def test_detects_suspicious_latin_token(self) -> None:
        answer = "Сумма POLITICOfgets бүтүндер и иных социальных выплат"
        assert is_garbage_answer(answer) is True

    def test_detects_password_and_repeated_tokens(self) -> None:
        answer = "К CEPASSWORDSF_castам банки предъявляют требования SFSFSFSFSF"
        assert is_garbage_answer(answer) is True

    def test_allows_known_banking_latin_terms(self) -> None:
        answer = "Проверьте SMS, IBAN, QR-код и 3-D Secure в интернет-банке."
        assert is_garbage_answer(answer) is False


class TestReferenceCleaning:
    """Tests for reference answer cleanup."""

    def test_strips_fragment_preamble(self) -> None:
        answer = "Согласно Фрагменту 2, уведомления приходят в SMS."
        assert clean_reference_answer(answer) == "уведомления приходят в SMS."

    def test_removes_chunk_markers(self) -> None:
        answer = "В Фрагмент 5: история платежей отображает магазин и бонусы."
        assert clean_reference_answer(answer) == "история платежей отображает магазин и бонусы."


class TestFinalizeGeneratedAnswer:
    """Tests for garbage rescue fallback."""

    def test_rescues_garbage_with_reference_answer(self) -> None:
        result = finalize_generated_answer(
            "Номер undergoingётного счёта можно узнать в личном кабинете",
            "Как узнать номер расчётного счёта?",
            "Номер счёта можно посмотреть в договоре.",
            "Номер расчётного счёта можно посмотреть в договоре банковского обслуживания.",
        )
        assert result == "Номер расчётного счёта можно посмотреть в договоре банковского обслуживания."

    def test_keeps_clean_llm_answer(self) -> None:
        result = finalize_generated_answer(
            "Номер счёта можно посмотреть в личном кабинете.",
            "Как узнать номер счёта?",
            "Номер счёта можно посмотреть в договоре.",
        )
        assert result == "Номер счёта можно посмотреть в личном кабинете."
