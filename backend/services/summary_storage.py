"""
summary_storage.py
==================
Saves and loads per-chunk summaries to disk so that if the Gemini API
fails mid-document (quota, network, etc.), the next run resumes from
where it stopped instead of starting over.

Storage location: backend/data/chunk_summaries.json
    The directory is defined in config.DATA_DIR so it is always
    consistent regardless of which folder you run scripts from.
"""

import json
import logging
import config

logger = logging.getLogger(__name__)

# Fixed path — always inside backend/data/
SUMMARY_FILE = config.DATA_DIR / "chunk_summaries.json"


def _ensure_data_dir() -> None:
    """Create the data directory if it does not exist."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_summary(chunk_number: int, summary: str) -> None:
    """
    Save or update the summary for a specific chunk number.

    Parameters
    ----------
    chunk_number : int   — 1-based chunk index
    summary : str        — the summary text to store
    """
    _ensure_data_dir()

    data = load_summaries()
    data[str(chunk_number)] = summary

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.debug("Saved summary for chunk %d to %s", chunk_number, SUMMARY_FILE)


def load_summaries() -> dict:
    """
    Load all previously saved chunk summaries.

    Returns
    -------
    dict  — keys are chunk numbers as strings ("1", "2", ...),
            values are summary strings.
            Returns an empty dict if no file exists yet.
    """
    if not SUMMARY_FILE.exists():
        return {}

    try:
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read summary file %s: %s", SUMMARY_FILE, e)
        return {}


def clear_summaries() -> None:
    """
    Delete all saved chunk summaries.

    Call this at the start of a new document upload so stale
    summaries from the previous document are not reused.
    """
    _ensure_data_dir()

    if SUMMARY_FILE.exists():
        SUMMARY_FILE.unlink()
        logger.info("Cleared chunk summaries from %s", SUMMARY_FILE)


def has_summary(chunk_number: int) -> bool:
    """
    Check whether a summary for a specific chunk already exists.

    Useful for skipping chunks that were already processed in a
    previous run (resume-on-failure logic).
    """
    return str(chunk_number) in load_summaries()