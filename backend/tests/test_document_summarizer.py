"""
Integration test for document summarizer in mock mode.
Ensures zero Gemini quota is consumed during unit tests.
"""

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


def test_summary_translation_uses_english_as_source(monkeypatch):
    monkeypatch.setattr(config, "USE_MOCK_SUMMARIZER", False)
    monkeypatch.setattr(
        document_summarizer,
        "summarize_text",
        lambda **kwargs: {
            "summary": "An English summary.",
            "mock": False,
            "error": "",
        },
    )
    translation_calls = []

    def fake_translate(text, target_language, source_language):
        translation_calls.append((text, target_language, source_language))
        return "தமிழ் சுருக்கம்."

    monkeypatch.setattr(document_summarizer, "translate_text", fake_translate)

    result = summarize_document(
        chunks=["Unique source content for translation test."],
        summary_length="short",
        target_language="Tamil",
        clear_cache=True,
    )

    assert result["summary"] == "தமிழ் சுருக்கம்."
    assert translation_calls == [
        ("An English summary.", "Tamil", "English")
    ]
