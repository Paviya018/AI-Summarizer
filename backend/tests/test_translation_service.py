from types import SimpleNamespace

import pytest

from services import translation_service
from services.translation_service import translate_text


def test_translation_uses_configured_provider_for_requested_language(monkeypatch):
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return SimpleNamespace(text="வணக்கம்")

    monkeypatch.setattr(translation_service, "generate_ai_content", fake_generate)

    assert translate_text("Hello", "Tamil", "English") == "வணக்கம்"
    assert "Translate the source text into Tamil" in prompts[0]
    assert "The source language is English." in prompts[0]


def test_translation_supports_all_languages_offered_by_app(monkeypatch):
    import config

    targets = []

    def fake_generate(prompt):
        targets.append(next(
            language
            for language in config.SUPPORTED_OUTPUT_LANGUAGES
            if f"into {language}." in prompt
        ))
        return SimpleNamespace(text="translated")

    monkeypatch.setattr(translation_service, "generate_ai_content", fake_generate)

    for language in config.SUPPORTED_OUTPUT_LANGUAGES:
        assert translate_text("Hello", language, "auto") == "translated"

    assert targets == list(config.SUPPORTED_OUTPUT_LANGUAGES)


def test_translation_auto_detects_source_language(monkeypatch):
    prompts = []
    monkeypatch.setattr(
        translation_service,
        "generate_ai_content",
        lambda prompt: prompts.append(prompt) or SimpleNamespace(text="translated"),
    )

    assert translate_text("Hello", "Tamil") == "translated"
    assert "Detect the source language" in prompts[0]


def test_translation_splits_long_text_to_gemini_limit(monkeypatch):
    translated_chunks = []

    def fake_generate(prompt):
        source = prompt.split("Source text:\n", 1)[1]
        translated_chunks.append(source.strip())
        return SimpleNamespace(text=f"translated-{len(translated_chunks)}")

    monkeypatch.setattr(translation_service, "generate_ai_content", fake_generate)
    text = "word " * 2000
    result = translate_text(text, "Tamil", "English")

    assert len(translated_chunks) > 1
    assert all(
        len(chunk) <= 8000
        for chunk in translation_service._split_for_provider(text)
    )
    assert result == "\n".join(
        f"translated-{index}" for index in range(1, len(translated_chunks) + 1)
    )


def test_ollama_translation_uses_smaller_chunks(monkeypatch):
    import config

    translated_chunks = []
    monkeypatch.setattr(config, "AI_PROVIDER", "ollama")

    def fake_generate(prompt):
        source = prompt.split("Source text:\n", 1)[1]
        translated_chunks.append(source.strip())
        return SimpleNamespace(text="translated")

    monkeypatch.setattr(translation_service, "generate_ai_content", fake_generate)

    translate_text("word " * 2000, "Tamil", "English")

    assert len(translated_chunks) > 2
    assert all(
        len(chunk) <= translation_service.OLLAMA_TRANSLATION_CHUNK_CHARS
        for chunk in translated_chunks
    )


def test_translation_rejects_unsupported_target_language():
    with pytest.raises(ValueError, match="Unsupported target language"):
        translate_text("Hello", "Klingon", "English")


def test_translation_does_not_call_provider_for_same_source_and_target(monkeypatch):
    monkeypatch.setattr(
        translation_service,
        "generate_ai_content",
        lambda prompt: (_ for _ in ()).throw(AssertionError("AI provider should not be called")),
    )

    assert translate_text("Hello", "English", "English") == "Hello"


def test_translation_reports_missing_provider_key(monkeypatch):
    monkeypatch.setattr(
        translation_service,
        "generate_ai_content",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("No AI provider API key is configured.")),
    )

    with pytest.raises(RuntimeError, match="AI translation failed"):
        translate_text("Hello", "Tamil", "English")


def test_empty_translation_returns_empty_without_calling_gemini(monkeypatch):
    monkeypatch.setattr(
        translation_service,
        "generate_ai_content",
        lambda prompt: (_ for _ in ()).throw(AssertionError("AI provider should not be called")),
    )

    assert translate_text("  ", "Tamil", "English") == ""
