"""
config.py
=========
Single source of truth for all application settings.

Every service module imports values from here instead of
reading os.getenv() or hardcoding values directly.

This ensures:
- Settings are validated once at startup.
- Changing a value (e.g. model name) only requires one edit.
- The .env file is loaded reliably before any other module runs.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------ #
# 1. Load the .env file
#    Path: backend/.env
#    dotenv_path resolves relative to this config.py file itself,
#    so it works correctly no matter where you launch the app from.
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parent   # = e:/AI-Document-Summarizer/backend/
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE)

# ------------------------------------------------------------------ #
# 2. AI provider settings
# ------------------------------------------------------------------ #
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OLLAMA_BASE_URL: str = os.getenv(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
).rstrip("/")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
AI_PROVIDER: str = os.getenv("AI_PROVIDER", "gemini").strip().lower()

# Maximum characters sent in a single AI API call.
MAX_CHARS_PER_CHUNK: int = 8_000

# Overlap between chunks so context is not lost at split points.
CHUNK_OVERLAP: int = 300

# ------------------------------------------------------------------ #
# 4. Summarizer mode
#    USE_MOCK_SUMMARIZER=true  → no API calls (safe during dev)
#    USE_MOCK_SUMMARIZER=false → configured AI providers
# ------------------------------------------------------------------ #
USE_MOCK_SUMMARIZER: bool = (
    os.getenv("USE_MOCK_SUMMARIZER", "false").lower() == "true"
)

# ------------------------------------------------------------------ #
# 5. File upload settings
# ------------------------------------------------------------------ #
ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png"}

# OCR language selected for images and scanned PDFs.
OCR_LANGUAGE_OPTIONS: dict[str, str] = {
    "en": "English",
    "latin": "French, German, or Spanish (Latin script)",
    "el": "Greek",
    "ta": "Tamil",
    "te": "Telugu",
    "devanagari": "Hindi",
    "ka": "Kannada",
}

MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ------------------------------------------------------------------ #
# 6. Directory paths
#    These folders are created automatically when the app starts.
# ------------------------------------------------------------------ #
UPLOADS_DIR: Path = BASE_DIR / "uploads"
AUDIO_DIR: Path = BASE_DIR / "generated_audio"
DATA_DIR: Path = BASE_DIR / "data"      # stores chunk_summaries.json etc.

# ------------------------------------------------------------------ #
# 7. Flask settings
# ------------------------------------------------------------------ #
FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
FLASK_DEBUG: bool = FLASK_ENV == "development"
SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-change-in-production")

# ------------------------------------------------------------------ #
# 8. Supported output languages for summaries
#    key  = display name shown in UI
#    value = language code used in prompts
# ------------------------------------------------------------------ #
SUPPORTED_OUTPUT_LANGUAGES: dict = {
    "English":    "English",
    "Tamil":      "Tamil",
    "Hindi":      "Hindi",
    "Telugu":     "Telugu",
    "Kannada":    "Kannada",
    "Malayalam":  "Malayalam",
    "French":     "French",
    "German":     "German",
    "Spanish":    "Spanish",
}

# ------------------------------------------------------------------ #
# 9. Summary length options
#    key  = option name shown in UI
#    value = instruction added to the AI prompt
# ------------------------------------------------------------------ #
SUMMARY_LENGTH_INSTRUCTIONS: dict = {
    "short":    "Write a concise summary in 2-3 sentences.",
    "medium":   "Write a clear summary in one paragraph (5-7 sentences).",
    "detailed": "Write a detailed summary covering all major points, "
                "preserving key facts, dates, numbers, and conclusions.",
}

# ------------------------------------------------------------------ #
# 10. OCR settings
# ------------------------------------------------------------------ #
# DPI multiplier when rendering PDF pages as images for OCR.
# 2 = 2× the default 72 DPI = 144 DPI.  Higher = slower but more accurate.
OCR_RENDER_SCALE: int = 2


def validate_config() -> list[str]:
    """
    Call this once at startup.
    Returns a list of warning messages for missing/invalid settings.
    Does NOT raise exceptions — the app can still run in mock mode
    even without an AI provider API key.
    """
    warnings = []

    has_gemini_key = GEMINI_API_KEY not in {"", "your_gemini_api_key_here"}
    if AI_PROVIDER != "ollama" and not has_gemini_key and not USE_MOCK_SUMMARIZER:
        warnings.append(
            "GEMINI_API_KEY is not set and "
            "USE_MOCK_SUMMARIZER is false. Summarization will fail. "
            "Set GEMINI_API_KEY in backend/.env, select AI_PROVIDER=ollama, or set "
            "USE_MOCK_SUMMARIZER=true to use mock mode."
        )

    if not ENV_FILE.exists():
        warnings.append(
            f".env file not found at {ENV_FILE}. "
            "Copy .env.example to .env and fill in your values."
        )

    return warnings
