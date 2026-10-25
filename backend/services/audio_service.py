"""
audio_service.py
================
Converts text into speech using language-specific Microsoft neural voices.
"""

import asyncio
import re
import uuid
import logging
import edge_tts
import config

logger = logging.getLogger(__name__)


def _is_network_tts_error(error: Exception) -> bool:
    """Return True when the failure is caused by outbound TTS connectivity issues."""
    message = str(error).lower()
    blocked_markers = [
        "speech.platform.bing.com",
        "getaddrinfo failed",
        "connection refused",
        "network is unreachable",
        "temporarily unavailable",
        "name or service not known",
        "timed out",
        "ssl",
        "certificate verify failed",
    ]
    return any(marker in message for marker in blocked_markers)


LANGUAGE_VOICE_MAP = {
    "english": "en-IN-NeerjaNeural",
    "tamil": "ta-IN-PallaviNeural",
    "hindi": "hi-IN-SwaraNeural",
    "telugu": "te-IN-ShrutiNeural",
    "kannada": "kn-IN-SapnaNeural",
    "malayalam": "ml-IN-SobhanaNeural",
    "french": "fr-FR-DeniseNeural",
    "german": "de-DE-KatjaNeural",
    "spanish": "es-ES-ElviraNeural",
}


async def _synthesize_chunk(text: str, voice: str, filepath) -> None:
    """Save one speech segment using its selected neural voice."""
    await edge_tts.Communicate(text=text, voice=voice).save(str(filepath))


async def _synthesize_chunks(chunks: list[str], voice: str, filepaths: list) -> None:
    """Synthesize a few segments concurrently while retaining output order."""
    semaphore = asyncio.Semaphore(4)

    async def synthesize_limited(text: str, filepath) -> None:
        async with semaphore:
            await _synthesize_chunk(text, voice, filepath)

    await asyncio.gather(
        *(synthesize_limited(text, filepath) for text, filepath in zip(chunks, filepaths))
    )


def _chunk_text(text: str, max_chunk_chars: int = 3000) -> list[str]:
    """Split speech text at sentence boundaries into bounded service requests."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chunk_chars])
            sentence = sentence[max_chunk_chars:]

        if len(current) + len(sentence) + bool(current) <= max_chunk_chars:
            current += (" " if current else "") + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks or [text]


def _clean_text_for_speech(text: str) -> str:
    """
    Cleans text for natural audio synthesis:
    - Strips markdown headers, bullet asterisks, bold/italic markers
    - Strips bracketed tags
    - Preserves sentence-ending periods, commas, and question marks for natural pauses
    """
    if not text:
        return ""

    # Remove headers (### Header)
    cleaned = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove bracketed tags like [MOCK ...]
    cleaned = re.sub(r"\[.*?\]", "", cleaned)
    # Remove markdown formatting (*, _, ~, `, #)
    cleaned = re.sub(r"[*_~`#]", "", cleaned)
    # Remove bullet symbols at the start of lines
    cleaned = re.sub(r"^\s*[-•*]\s+", "", cleaned, flags=re.MULTILINE)
    # Replace multiple spaces or newlines with a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def generate_audio(text: str, language: str = "English", audio_id: str = None, is_full_document: bool = False) -> dict:
    """
    Generates an MP3 audio file using a native-language neural voice.
    If is_full_document is True, processes long texts by chunking and concatenating
    MP3 frames to produce a full audiobook-style recording without hitting length limits.
    """
    if not text or not text.strip():
        return {
            "success": False,
            "filename": "",
            "filepath": "",
            "language": language,
            "error": "Cannot generate audio for empty text."
        }

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    lang_lower = language.strip().lower()
    voice = LANGUAGE_VOICE_MAP.get(lang_lower)
    if voice is None:
        return {
            "success": False,
            "filename": "",
            "filepath": "",
            "language": language,
            "error": f"Speech synthesis is not configured for language '{language}'.",
        }

    clean_text = _clean_text_for_speech(text)
    if not clean_text:
        return {
            "success": False,
            "filename": "",
            "filepath": "",
            "language": language,
            "error": "No speakable text remained after preprocessing."
        }

    file_prefix = "full_audio" if is_full_document else "audio"
    file_id = audio_id or uuid.uuid4().hex[:12]
    filename = f"{file_prefix}_{file_id}.mp3"
    filepath = config.AUDIO_DIR / filename

    # --- Mode 1: Summary mode (Standard <= 3500 chars) ---
    if not is_full_document:
        if len(clean_text) > 3500:
            truncated = clean_text[:3500]
            last_period = truncated.rfind(".")
            clean_text = (truncated[: last_period + 1] if last_period > 1000 else truncated + "...")

        try:
            logger.info("Generating summary audio: lang='%s' (voice: %s, chars: %d)...",
                        language, voice, len(clean_text))
            asyncio.run(_synthesize_chunk(clean_text, voice, filepath))
            return {
                "success": True,
                "filename": filename,
                "filepath": str(filepath),
                "language": language,
                "error": "",
            }
        except Exception as e:
            logger.exception("Summary audio generation failed: %s", e)
            if _is_network_tts_error(e):
                return {
                    "success": False,
                    "filename": "",
                    "filepath": "",
                    "language": language,
                    "error": (
                        "Text-to-speech service is unavailable because the app cannot reach the Microsoft Edge speech service. "
                        "Check your internet connection, VPN, proxy, or firewall settings, then try again."
                    ),
                }
            return {"success": False, "filename": "", "filepath": "", "language": language, "error": str(e)}

    # --- Mode 2: Full Document Audio (Concatenate chunks) ---
    try:
        logger.info("Generating Full Document Audio: lang='%s' (voice: %s, total chars: %d)...",
                    language, voice, len(clean_text))
        
        # Larger batches reduce network round trips; keep them below service limits.
        chunks = _chunk_text(clean_text)

        temp_files = []
        for idx in range(len(chunks)):
            temp_path = config.AUDIO_DIR / f"temp_{file_id}_{idx}.mp3"
            temp_files.append(temp_path)
        try:
            logger.info(
                "Synthesizing full audio in %d chunks (up to %d concurrent requests).",
                len(chunks), 4,
            )
            asyncio.run(_synthesize_chunks(chunks, voice, temp_files))

            # Append MP3 segments so long documents are not truncated to one request.
            with open(filepath, "wb") as outfile:
                for temp_path in temp_files:
                    with open(temp_path, "rb") as infile:
                        outfile.write(infile.read())
        finally:
            for temp_path in temp_files:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary audio segment: %s", temp_path)

        logger.info("Full document audio generated successfully: %s (%d bytes)", filename, filepath.stat().st_size)
        return {
            "success": True,
            "filename": filename,
            "filepath": str(filepath),
            "language": language,
            "error": "",
        }

    except Exception as e:
        logger.exception("Full document audio generation failed: %s", e)
        if _is_network_tts_error(e):
            return {
                "success": False,
                "filename": "",
                "filepath": "",
                "language": language,
                "error": (
                    "Text-to-speech service is unavailable because the app cannot reach the Microsoft Edge speech service. "
                    "Check your internet connection, VPN, proxy, or firewall settings, then try again."
                ),
            }
        return {
            "success": False,
            "filename": "",
            "filepath": "",
            "language": language,
            "error": f"Failed to generate full document audio: {str(e)}",
        }
