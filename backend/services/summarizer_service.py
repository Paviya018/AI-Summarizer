"""
summarizer_service.py
=====================
Handles Gemini inference with local Ollama fallback.

Key design decisions:
- Imports config first so .env is loaded before anything else runs.
- Does NOT crash at import time if the API key is missing.
  Instead it logs a warning and the mock fallback takes over.
- Accepts summary_length and target_language so the caller controls
  output style — no hardcoded prompt text.
- Returns structured dicts so the caller always knows whether the
  result came from the real API or the mock.
"""

import config  # Must be first — loads .env before os.getenv() calls

import logging
import time
from types import SimpleNamespace

import requests
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from services.mock_summarizer import mock_summarize

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Gemini client is created once at module load.
# ------------------------------------------------------------------ #
_client = None


def _has_api_key(value: str, placeholder: str) -> bool:
    return bool(value and value != placeholder)


def _configured_providers() -> list[str]:
    """Return configured providers in preferred-first failover order."""
    preferred = config.AI_PROVIDER
    available = {
        "gemini": _has_api_key(
            config.GEMINI_API_KEY, "your_gemini_api_key_here"
        ),
        "ollama": True,
    }
    if preferred not in available:
        logger.warning(
            "Unknown AI_PROVIDER '%s'; using Gemini then Ollama.",
            preferred,
        )
        preferred = "gemini"

    if preferred == "ollama":
        return ["ollama"]

    return [
        name
        for name in (preferred, "ollama")
        if available[name]
    ]


def _get_client():
    """
    Lazily creates the Gemini client.
    Returns None if the API key is not set.
    """
    global _client

    if _client is not None:
        return _client

    if not _has_api_key(config.GEMINI_API_KEY, "your_gemini_api_key_here"):
        return None

    _client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=genai_types.HttpOptions(
            timeout=30_000,
            retry_options=genai_types.HttpRetryOptions(attempts=1),
        ),
    )
    return _client


def _generate_content_with_retry(client, model: str, prompt: str):
    """Retry brief Gemini service-unavailable responses before using fallback."""
    for attempt in range(3):
        try:
            return client.models.generate_content(model=model, contents=prompt)
        except genai_errors.APIError as e:
            if e.code != 503 or attempt == 2:
                raise
            delay = 2 ** attempt
            logger.warning(
                "Gemini is temporarily unavailable; retrying in %d second(s).",
                delay,
            )
            time.sleep(delay)


def _generate_ollama_content(prompt: str):
    """Generate content with the local Ollama HTTP API."""
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": f"/no_think\n\n{prompt}",
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": 4096,
                "num_predict": 1536,
                "temperature": 0.2,
                "repeat_penalty": 1.2,
                "repeat_last_n": 128,
            },
        },
        timeout=(10, 600),
    )
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(
            f"Ollama request failed (HTTP {response.status_code}): {detail}"
        )

    text = response.json().get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return SimpleNamespace(text=text)


def generate_ai_content(prompt: str):
    """Generate content with the first working configured AI provider."""
    providers = _configured_providers()
    if not providers:
        raise RuntimeError(
            "No AI provider is available. Select Ollama or set GEMINI_API_KEY "
            "in backend/.env."
        )

    errors = []
    for provider in providers:
        try:
            if provider == "ollama":
                response = _generate_ollama_content(prompt)
            else:
                client = _get_client()
                if client is None:
                    raise RuntimeError("GEMINI_API_KEY is not configured.")
                response = _generate_content_with_retry(
                    client, config.GEMINI_MODEL, prompt
                )
            logger.info("AI request succeeded with %s.", provider.capitalize())
            return response
        except Exception as error:
            errors.append(f"{provider}: {error}")
            logger.warning(
                "%s request failed; trying the next configured AI provider.",
                provider.capitalize(),
                exc_info=True,
            )

    raise RuntimeError("All configured AI providers failed: " + "; ".join(errors))


def generate_gemini_content(prompt: str):
    """Backward-compatible alias for existing callers."""
    return generate_ai_content(prompt)


def _mock_result(text: str, label: str = "MOCK SUMMARY") -> dict:
    summary = mock_summarize(text)
    return {"summary": f"[{label}] {summary}", "mock": True, "error": ""}


def _format_provider_error(error: Exception, source_text: str) -> dict:
    error_msg = str(error)
    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
        logger.warning("All configured AI providers exhausted their quota.")
        return _mock_result(source_text, "MOCK SUMMARY - QUOTA FALLBACK")
    formatted_err = f"AI provider error ({type(error).__name__}): {error}"
    logger.error(formatted_err)
    return {"summary": "", "mock": False, "error": formatted_err}


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def summarize_text(
    text: str,
    summary_length: str = "medium",
    target_language: str = "English",
) -> dict:
    """
    Summarize a block of text using Gemini, Ollama, or the mock fallback.

    Parameters
    ----------
    text : str
        The text to summarize.
    summary_length : str
        One of "short", "medium", or "detailed".
        Controls how long the summary is.
    target_language : str
        The language in which the summary should be written,
        e.g. "English", "Tamil", "Hindi".

    Returns
    -------
    dict with keys:
        summary  (str)  — the generated summary text
        mock     (bool) — True if this came from the mock summarizer
        error    (str)  — non-empty string if something went wrong
    """

    # ---- Guard: empty input ----------------------------------------
    if not text or not text.strip():
        return {"summary": "", "mock": False, "error": ""}

    # ---- Decide whether to use mock --------------------------------
    use_mock = config.USE_MOCK_SUMMARIZER or not _configured_providers()

    if use_mock:
        logger.info("Using mock summarizer.")
        summary = mock_summarize(text)
        return {
            "summary": f"[MOCK SUMMARY] {summary}",
            "mock": True,
            "error": "",
        }

    # ---- Build the prompt ------------------------------------------
    length_instruction = config.SUMMARY_LENGTH_INSTRUCTIONS.get(
        summary_length,
        config.SUMMARY_LENGTH_INSTRUCTIONS["medium"],   # safe default
    )

    prompt = f"""You are an expert document summarizer.

Task: {length_instruction}

Output language: Write the entire summary in {target_language}.

Rules:
- Do NOT invent facts, dates, or numbers that are not in the source text.
- Preserve all technical terms, proper names, dates, and numerical values exactly.
- Do not repeat the same information.
- Do not add a title or heading to your response.
- Return only the summary text, nothing else.

Source text:
{text}
"""

    # ---- Call configured AI providers with automatic failover -------
    try:
        logger.info(
            "Calling AI providers | primary=%s | length=%s | language=%s",
            config.AI_PROVIDER, summary_length, target_language,
        )

        response = generate_ai_content(prompt)

        summary = response.text.strip()

        logger.info("AI response received (%d chars).", len(summary))

        return {"summary": summary, "mock": False, "error": ""}

    except Exception as e:
        return _format_provider_error(e, text)


def summarize_chunks(
    chunks: list[str],
    summary_length: str = "medium",
    target_language: str = "English",
) -> dict:
    """
    Summarize a list of text chunks and combine into a final summary.

    Used when a document is too long to send in one API call.

    Steps:
    1. Summarize each chunk individually.
    2. Combine the chunk summaries.
    3. Generate one final coherent summary from the combined text.

    Returns
    -------
    dict with keys:
        summary       (str)        — the final combined summary
        chunk_count   (int)        — how many chunks were processed
        mock          (bool)       — True if mock mode was used
        errors        (list[str])  — any per-chunk errors
    """

    if not chunks:
        return {
            "summary": "",
            "chunk_count": 0,
            "mock": False,
            "errors": [],
        }

    chunk_summaries = []
    errors = []
    any_mock = False

    for i, chunk in enumerate(chunks, start=1):
        logger.info("Summarizing chunk %d / %d ...", i, len(chunks))

        result = summarize_text(
            text=chunk,
            summary_length="medium",       # chunk summaries always medium
            target_language="English",     # always combine in English first
        )

        if result["error"]:
            errors.append(f"Chunk {i}: {result['error']}")
            logger.warning("Chunk %d failed: %s", i, result["error"])
            continue

        if result["mock"]:
            any_mock = True

        if result["summary"]:
            chunk_summaries.append(result["summary"])

    if not chunk_summaries:
        return {
            "summary": "",
            "chunk_count": len(chunks),
            "mock": any_mock,
            "errors": errors or ["All chunks failed to summarize."],
        }

    # ---- If only one chunk, return its summary directly ------------
    if len(chunk_summaries) == 1:
        return {
            "summary": chunk_summaries[0],
            "chunk_count": 1,
            "mock": any_mock,
            "errors": errors,
        }

    # ---- Combine chunk summaries into one final summary ------------
    logger.info("Combining %d chunk summaries into final summary...", len(chunk_summaries))

    combined = "\n\n".join(
        f"Section {i}:\n{s}" for i, s in enumerate(chunk_summaries, 1)
    )

    final_prompt = f"""You are combining section summaries of a large document into one final summary.

Task: {config.SUMMARY_LENGTH_INSTRUCTIONS.get(summary_length, config.SUMMARY_LENGTH_INSTRUCTIONS['medium'])}

Output language: Write the entire summary in {target_language}.

Rules:
- Merge the sections into ONE coherent paragraph or set of paragraphs.
- Remove repetition across sections.
- Preserve all technical terms, dates, names, and numbers.
- Do NOT invent anything not present in the sections below.
- Do not add a title or heading.
- Return only the summary text.

Section summaries:
{combined}
"""

    final_result = summarize_text(
        text=final_prompt,
        summary_length=summary_length,
        target_language=target_language,
    )

    return {
        "summary": final_result["summary"],
        "chunk_count": len(chunks),
        "mock": any_mock or final_result["mock"],
        "errors": errors + ([final_result["error"]] if final_result["error"] else []),
    }