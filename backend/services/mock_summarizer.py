"""
mock_summarizer.py
==================
A high-quality content-aware extractive summarizer.

Filters out:
- Figure captions (Figure 1.1, Fig B.2, etc.)
- Table captions (Table 3.1, etc.)
- Appendix and Screenshot labels
- Table of Contents entries
- Bullet fragments and page numbers

Selects:
- Complete grammatical sentences describing actual objectives, methods, architecture, and results.
"""

import re
from collections import Counter

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "this", "that", "these", "those", "it", "its", "as", "if", "each",
    "than", "then", "very", "much", "more", "also", "such"
}

# Patterns that indicate document boilerplate, captions, or appendix items
JUNK_PATTERNS = [
    r"^(fig|figure|table|chart|graph|diagram)\s*[\d\.\w]*\s*[:\-\.]",  # Figure 1.1: ...
    r"^(appendix|annexure|screenshot|exhibit)\b",                       # Appendix B: ...
    r"^(chapter|section)\s*[\d\.\w]+",                                  # Chapter 1
    r"^\s*[\d\.]+\s+[A-Z]",                                             # 1.2 Introduction
    r"^page\s*\d+",                                                     # Page 12
    r"^submitted\s+by",                                                 # Submitted by ...
    r"^under\s+the\s+guidance",                                         # Under the guidance of ...
    r"^department\s+of",                                                # Department of CSE
    r"^\s*[\(\[]?\s*[a-z0-9]\s*[\)\]\.]\s*$",                          # (a), 1., etc.
]


def _is_boilerplate_or_caption(sentence: str) -> bool:
    """Returns True if the line is a caption, TOC entry, or boilerplate."""
    s_lower = sentence.strip().lower()

    # Too short to be an informative summary sentence
    if len(s_lower.split()) < 8:
        return True

    # Caption / heading patterns
    for pat in JUNK_PATTERNS:
        if re.search(pat, s_lower):
            return True

    # Lines ending with page numbers or trailing digits
    if re.search(r"\b\d{1,3}\s*$", s_lower):
        return True

    # Must contain at least one common verb to be a real sentence
    common_verbs = ["is", "are", "was", "were", "has", "have", "had", "can", "uses",
                    "provides", "proposes", "implements", "developed", "achieved",
                    "presents", "consists", "enables", "detects", "classifies"]
    words = set(re.findall(r"\b[a-z]+\b", s_lower))
    if not any(v in words for v in common_verbs):
        return True

    return False


def _score_sentences(sentences: list[str]) -> list[tuple[float, str]]:
    """Scores sentences based on word frequency and informative key phrases."""
    all_words = []
    for s in sentences:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", s.lower())
        all_words.extend([w for w in tokens if w not in STOP_WORDS])

    freq = Counter(all_words)
    max_freq = max(freq.values(), default=1)

    scored = []
    for s in sentences:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", s.lower())
        meaningful = [w for w in tokens if w not in STOP_WORDS]

        if len(meaningful) < 5 or len(s.split()) > 50:
            continue

        score = sum(freq[w] / max_freq for w in meaningful)
        score = score / (len(meaningful) ** 0.5)

        # High priority for informative findings and project definitions
        s_lower = s.lower()
        if any(term in s_lower for term in ["proposes", "implements", "achieved", "accuracy", "accuracy of", "pipeline", "architecture", "results show"]):
            score *= 1.4

        scored.append((score, s))

    return scored


def mock_summarize(text: str, length: str = "medium") -> str:
    """
    Extracts an informative summary of real sentences from across the whole document.
    """
    if not text or not text.strip():
        return ""

    # Split into candidate sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Keep only genuine narrative sentences
    clean_sentences = [
        s.strip() for s in raw_sentences
        if not _is_boilerplate_or_caption(s)
    ]

    if not clean_sentences:
        # Fallback to general sentences
        clean_sentences = [s.strip() for s in raw_sentences if len(s.split()) >= 8]

    if not clean_sentences:
        return text[:400].strip() + "..."

    target_count = {
        "short": 3,
        "medium": 6,
        "detailed": 10,
    }.get(length, 6)

    scored = _score_sentences(clean_sentences)
    if not scored:
        return " ".join(clean_sentences[:target_count])

    top_scored = sorted(scored, key=lambda x: x[0], reverse=True)[:target_count]
    top_texts = {s for _, s in top_scored}

    # Maintain original document order
    ordered = [s for s in clean_sentences if s in top_texts]

    # Clean double spaces or broken formatting
    result = " ".join(ordered)
    result = re.sub(r"\s+", " ", result).strip()

    return result