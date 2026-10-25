"""
translation_service.py
======================
Translates text with Gemini/Ollama provider failover.
"""

import logging
import re

import config

from services.summarizer_service import generate_ai_content

logger = logging.getLogger(__name__)

OLLAMA_TRANSLATION_CHUNK_CHARS = 4_000


def _split_for_provider(text: str, max_chars: int | None = None) -> list[str]:
    """Split long text into word-safe requests within provider input limits."""
    limit = max_chars or config.MAX_CHARS_PER_CHUNK
    chunks = []
    current = ""

    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for word in sentence.split():
            while len(word) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(word[:limit])
                word = word[limit:]
            if len(current) + len(word) + bool(current) <= limit:
                current += (" " if current else "") + word
            else:
                chunks.append(current)
                current = word

    if current:
        chunks.append(current)
    return chunks or [text]


def _translate_chunk(
    text: str,
    target_language: str,
    source_language: str,
) -> str:
    source_instruction = (
        f"The source language is {source_language}."
        if source_language.strip().lower() not in {"", "auto", "unknown"}
        else "Detect the source language from the text."
    )
    prompt = f"""Translate the source text into {target_language}.

{source_instruction}

Rules:
- Preserve the original meaning, names, numbers, dates, and technical terms.
- Do not summarize, explain, or add information.
- Keep paragraph structure where practical.
- Return only the translated text.

Source text:
{text}
"""

    response = generate_ai_content(prompt)
    translated = (response.text or "").strip()
    if not translated:
        raise RuntimeError("AI provider returned an empty translation.")
    return translated


def translate_text(
    text: str,
    target_language: str = "English",
    source_language: str = "auto",
) -> str:
    """Translate text into the requested language using configured providers."""
    if not text or not text.strip():
        return ""

    target_lang_clean = target_language.strip().lower()
    source_lang_clean = source_language.strip().lower()
    if source_lang_clean == target_lang_clean and source_lang_clean not in {
        "",
        "auto",
        "unknown",
    }:
        return text

    supported_languages = {
        language.lower() for language in config.SUPPORTED_OUTPUT_LANGUAGES
    }
    if target_lang_clean not in supported_languages:
        raise ValueError(f"Unsupported target language: {target_language}")

    clean_input = re.sub(r"\[MOCK SUMMARY.*?\]\s*", "", text).strip()
    translated_chunks = []

    try:
        max_chars = (
            OLLAMA_TRANSLATION_CHUNK_CHARS
            if config.AI_PROVIDER == "ollama"
            else config.MAX_CHARS_PER_CHUNK
        )
        chunks = _split_for_provider(clean_input, max_chars=max_chars)
        for index, chunk in enumerate(chunks, start=1):
            logger.info(
                "Translating chunk %d/%d from '%s' to '%s'.",
                index,
                len(chunks),
                source_language,
                target_language,
            )
            translated_chunks.append(
                _translate_chunk(chunk, target_language, source_language)
            )
    except RuntimeError as error:
        logger.error("AI translation failed: %s", error)
        raise RuntimeError(f"AI translation failed: {error}") from error
    except Exception as error:
        logger.exception("Unexpected error during AI translation.")
        raise RuntimeError(f"AI translation failed: {error}") from error

    return "\n".join(translated_chunks)
