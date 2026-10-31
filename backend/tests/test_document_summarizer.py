"""
Integration test for document summarizer in mock mode.
Ensures zero Gemini quota is consumed during unit tests.
"""

import pytest

import config
from services.document_summarizer import summarize_document
from services import document_summarizer


def test_summarize_document_mock_mode(monkeypatch):
    # Force mock summarizer regardless of env settings
    monkeypatch.setattr(config, "USE_MOCK_SUMMARIZER", True)

    chunks = [
        "First chunk with several important sentences about technology.",
        "Second chunk detailing how algorithms and datasets operate.",
    ]

    result = summarize_document(
        chunks=chunks,
        summary_length="short",
        target_language="English",
        clear_cache=True,
    )

    assert result["mock"] is True
    assert result["chunk_count"] == 2
    assert result["summary"].strip()
    assert "[MOCK SUMMARY]" not in result["summary"]
    assert len(result["errors"]) == 0


def test_summary_is_translated_from_english_to_the_selected_language(monkeypatch):
    monkeypatch.setattr(config, "USE_MOCK_SUMMARIZER", False)
    summarize_calls = []

    def fake_summarize(**kwargs):
        summarize_calls.append(kwargs)
        return {
            "summary": "Artificial intelligence is a branch of computer science.",
            "mock": False,
            "error": "",
        }

    monkeypatch.setattr(document_summarizer, "summarize_text", fake_summarize)
    translation_calls = []
    monkeypatch.setattr(
        document_summarizer,
        "translate_text",
        lambda text, target_language, source_language: (
            translation_calls.append((text, target_language, source_language))
            or "செயற்கை நுண்ணறிவு கணினி அறிவியலின் ஒரு பிரிவாகும்."
        ),
    )

    result = summarize_document(
        chunks=["Unique source content for translation test."],
        summary_length="short",
        target_language="Tamil",
        clear_cache=True,
    )

    assert result["summary"] == "செயற்கை நுண்ணறிவு கணினி அறிவியலின் ஒரு பிரிவாகும்."
    assert summarize_calls[0]["target_language"] == "English"
    assert translation_calls == [
        (
            "Artificial intelligence is a branch of computer science.",
            "Tamil",
            "English",
        )
    ]


def test_combined_summary_uses_the_selected_language(monkeypatch):
    monkeypatch.setattr(config, "USE_MOCK_SUMMARIZER", False)
    summarize_calls = []

    def fake_summarize(**kwargs):
        summarize_calls.append(kwargs)
        return {"summary": "English section summary.", "mock": False, "error": ""}

    monkeypatch.setattr(document_summarizer, "summarize_text", fake_summarize)
    translation_calls = []
    monkeypatch.setattr(
        document_summarizer,
        "translate_text",
        lambda text, target_language, source_language: (
            translation_calls.append((text, target_language, source_language))
            or "தமிழ் ஆவணச் சுருக்கம்."
        ),
    )

    result = summarize_document(
        chunks=["First document section.", "Second document section."],
        summary_length="short",
        target_language="Tamil",
        clear_cache=True,
    )

    assert result["summary"] == "தமிழ் ஆவணச் சுருக்கம்."
    assert [call["target_language"] for call in summarize_calls] == [
        "English",
        "English",
        "English",
    ]
    assert translation_calls == [
        ("English section summary.", "Tamil", "English")
    ]


def test_mock_summary_is_translated_to_the_selected_language(monkeypatch):
    monkeypatch.setattr(config, "USE_MOCK_SUMMARIZER", True)
    translation_calls = []
    monkeypatch.setattr(
        document_summarizer,
        "translate_text",
        lambda text, target_language, source_language: (
            translation_calls.append((text, target_language, source_language))
            or "தமிழ் சுருக்கம்."
        ),
    )

    result = summarize_document(
        chunks=["The system provides reliable document processing capabilities."],
        summary_length="short",
        target_language="Tamil",
        clear_cache=True,
    )

    assert result["summary"] == "தமிழ் சுருக்கம்."
    assert translation_calls[0][1:] == ("Tamil", "English")


def test_summarize_document_rejects_unsupported_target_language():
    with pytest.raises(ValueError, match="Unsupported target language"):
        summarize_document(
            chunks=["A source document."],
            target_language="Klingon",
        )
