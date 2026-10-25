"""
document_summarizer.py
======================
Orchestrates the full document summarization pipeline.
Extracts representation across the entire document.
"""

import logging
import re
import config
from services.summarizer_service import summarize_text
from services.mock_summarizer import mock_summarize
from services.translation_service import translate_text
from services.summary_storage import (
    save_summary,
    load_summaries,
    clear_summaries,
)

logger = logging.getLogger(__name__)


def _clean_summary_text(text: str) -> str:
    """Removes mock headers and internal markers."""
    return re.sub(r"\[MOCK SUMMARY.*?\]\s*", "", text).strip()


def summarize_document(
    chunks: list[str],
    summary_length: str = "medium",
    target_language: str = "English",
    clear_cache: bool = True,
) -> dict:
    """
    Summarize a list of text chunks into a single coherent document summary
    covering the entire document from start to finish.
    """
    if not chunks:
        return {
            "summary": "",
            "chunk_count": 0,
            "mock": False,
            "errors": ["No text chunks provided."],
            "from_cache": 0,
        }

    if clear_cache:
        clear_summaries()

    saved = load_summaries()
    chunk_summaries = []
    errors = []
    any_mock = False
    from_cache = 0

    # 1. Summarize each chunk across the entire document
    for i, chunk in enumerate(chunks, start=1):
        if str(i) in saved:
            chunk_summaries.append(saved[str(i)])
            from_cache += 1
            continue

        result = summarize_text(
            text=chunk,
            summary_length="medium",
            target_language="English",
        )

        if result.get("error"):
            errors.append(f"Chunk {i}: {result['error']}")
            mock_s = mock_summarize(chunk, length="short")
            chunk_summaries.append(mock_s)
            save_summary(i, mock_s)
            any_mock = True
            continue

        if result.get("mock"):
            any_mock = True

        raw_summary = _clean_summary_text(result["summary"])
        if raw_summary:
            chunk_summaries.append(raw_summary)
            save_summary(i, raw_summary)

    if not chunk_summaries:
        return {
            "summary": "",
            "chunk_count": len(chunks),
            "mock": any_mock,
            "errors": errors or ["All chunks failed to produce a summary."],
            "from_cache": from_cache,
        }

    # 2. Combine all chunk summaries across the whole document
    if len(chunk_summaries) == 1:
        base_summary = chunk_summaries[0]
    else:
        # If in mock or quota fallback mode, extract key sentences across all chunks
        if any_mock or config.USE_MOCK_SUMMARIZER:
            whole_document_text = "\n\n".join(chunk_summaries)
            base_summary = mock_summarize(whole_document_text, length=summary_length)
        else:
            combined = "\n\n".join(
                f"Section {i}:\n{s}" for i, s in enumerate(chunk_summaries, 1)
            )
            length_instruction = config.SUMMARY_LENGTH_INSTRUCTIONS.get(
                summary_length, config.SUMMARY_LENGTH_INSTRUCTIONS["medium"]
            )
            final_prompt = f"""You are combining section summaries of a large document into one final summary.

Task: {length_instruction}

Rules:
- Cover the entire document from start to finish.
- Merge all sections into ONE coherent response.
- Remove repetition.
- Preserve key findings, numbers, and conclusions.
- Return only the summary text in English.

Section summaries:
{combined}
"""
            final_res = summarize_text(
                text=final_prompt,
                summary_length=summary_length,
                target_language="English",
            )
            if final_res.get("summary") and not final_res.get("mock"):
                base_summary = final_res["summary"]
            else:
                any_mock = True
                whole_document_text = "\n\n".join(chunk_summaries)
                base_summary = mock_summarize(whole_document_text, length=summary_length)

    # 3. Translate to requested target language
    final_text = base_summary
    if target_language.lower() != "english":
        logger.info("Translating full summary to %s...", target_language)
        translated = translate_text(
            base_summary,
            target_language=target_language,
            source_language="English",
        )
        if translated and translated.strip():
            final_text = translated

    return {
        "summary": final_text,
        "chunk_count": len(chunks),
        "mock": any_mock,
        "errors": errors,
        "from_cache": from_cache,
    }