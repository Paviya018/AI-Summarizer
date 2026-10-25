"""
chunk_service.py
================
Splits long text into chunks that respect paragraph and sentence
boundaries so that each chunk sent to Gemini is complete and coherent.

Why boundaries matter:
    Cutting text mid-sentence gives the AI an incomplete thought,
    which degrades summary quality. By splitting at paragraph breaks
    first, then sentence ends, every chunk is self-contained.

Strategy:
    1. Split the full text into paragraphs (double newlines).
    2. Keep adding paragraphs to the current chunk until adding the
       next paragraph would exceed max_chars.
    3. If a single paragraph is itself too long, split it at the
       nearest sentence end (. ! ?) before the limit.
    4. Overlap: the last sentence(s) of each chunk are repeated at
       the start of the next so context is never lost at the boundary.
"""

import re
import config


def _split_into_sentences(text: str) -> list[str]:
    """
    Split a block of text into individual sentences.
    Handles common abbreviations to avoid false splits (Mr., Dr., etc.)
    """
    # Sentence boundary: ends with . ! ? followed by whitespace + capital
    # Using a simple but reliable pattern
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z\u0080-\uFFFF])')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """
    Break a single oversized paragraph into sentence-boundary chunks.
    Called only when one paragraph alone exceeds max_chars.
    """
    sentences = _split_into_sentences(paragraph)
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip()

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single sentence is still too long, hard-cut it
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def split_text(
    text: str,
    max_chars: int = None,
    overlap_chars: int = None,
) -> list[str]:
    """
    Split text into chunks that respect paragraph and sentence boundaries.

    Parameters
    ----------
    text : str
        The full cleaned document text.
    max_chars : int, optional
        Maximum characters per chunk. Defaults to config.MAX_CHARS_PER_CHUNK.
    overlap_chars : int, optional
        How many characters from the end of each chunk to repeat at the
        start of the next. Defaults to config.CHUNK_OVERLAP.

    Returns
    -------
    list[str]
        Ordered list of text chunks. Each chunk is a complete set of
        paragraphs or sentences, never cut mid-sentence.
    """

    if not text or not text.strip():
        return []

    max_chars = config.MAX_CHARS_PER_CHUNK if max_chars is None else max_chars
    overlap_chars = config.CHUNK_OVERLAP if overlap_chars is None else overlap_chars

    # ---- Step 1: split into paragraphs ----------------------------
    # A paragraph is separated by one or more blank lines.
    paragraphs = re.split(r'\n{2,}', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # If the entire document fits in one chunk, return it as-is
    if len(text) <= max_chars:
        return [text.strip()]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        # If this single paragraph is too big, split it by sentence
        if len(paragraph) > max_chars:
            # Flush what we have so far
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Split the big paragraph by sentence
            sub_chunks = _split_long_paragraph(paragraph, max_chars)
            chunks.extend(sub_chunks)
            continue

        # Would adding this paragraph exceed the limit?
        candidate = (current_chunk + "\n\n" + paragraph).strip()

        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            # Save current chunk and start a new one
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Carry over the last overlap_chars of the previous chunk
            # to preserve context at the boundary
            if overlap_chars > 0 and current_chunk:
                overlap_text = current_chunk[-overlap_chars:]
                # Find the first sentence boundary in the overlap
                match = re.search(r'(?<=[.!?])\s+', overlap_text)
                if match:
                    overlap_text = overlap_text[match.end():]
                current_chunk = (overlap_text + "\n\n" + paragraph).strip()
            else:
                current_chunk = paragraph

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_summary(chunks: list[str]) -> dict:
    """
    Return basic statistics about the chunks for logging / debugging.
    """
    if not chunks:
        return {"count": 0, "min_chars": 0, "max_chars": 0, "total_chars": 0}

    sizes = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "total_chars": sum(sizes),
    }