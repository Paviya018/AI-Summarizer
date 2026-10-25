"""
Unit tests for chunk_service.py
"""

from services.chunk_service import split_text, chunk_summary


def test_split_text_empty():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_short():
    text = "Short single paragraph document."
    chunks = split_text(text, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_paragraphs():
    p1 = "First sentence of paragraph one. Second sentence."
    p2 = "First sentence of paragraph two. Second sentence."
    doc = f"{p1}\n\n{p2}"

    chunks = split_text(doc, max_chars=len(p1) + 10, overlap_chars=0)
    assert len(chunks) == 2
    assert chunks[0] == p1
    assert chunks[1] == p2


def test_chunk_summary_stats():
    chunks = ["Hello world.", "This is another chunk."]
    stats = chunk_summary(chunks)
    assert stats["count"] == 2
    assert stats["total_chars"] == sum(len(c) for c in chunks)
