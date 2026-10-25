"""
Unit tests for audio_service.py
"""

import asyncio
from collections import Counter
from services import audio_service
from services.audio_service import _clean_text_for_speech, generate_audio
import os


def test_clean_text_for_speech():
    raw = "### Title\n* Point one\n* Point two\n[MOCK SUMMARY] This is **bold** text."
    cleaned = _clean_text_for_speech(raw)
    assert "#" not in cleaned
    assert "*" not in cleaned
    assert "[MOCK SUMMARY]" not in cleaned
    assert "Title Point one Point two This is bold text." == cleaned


def test_generate_audio_empty():
    res = generate_audio("", "English")
    assert not res["success"]
    assert "empty text" in res["error"]


def test_generate_audio_uses_native_neural_voice(monkeypatch):
    generated = []

    async def fake_synthesize(text, voice, filepath):
        generated.append((text, voice))
        with open(filepath, "wb") as audio_file:
            audio_file.write(b"fake mp3")

    monkeypatch.setattr(audio_service, "_synthesize_chunk", fake_synthesize)
    res = generate_audio("Testing audio synthesis.", "English")
    assert res["success"]
    assert generated == [("Testing audio synthesis.", "en-IN-NeerjaNeural")]
    assert res["filename"].endswith(".mp3")
    assert os.path.exists(res["filepath"])
    assert os.path.getsize(res["filepath"]) > 0
    os.remove(res["filepath"])


def test_generate_full_audio_preserves_language_and_all_chunks(monkeypatch):
    generated = []
    active = 0
    max_active = 0

    async def fake_synthesize(text, voice, filepath):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        generated.append((text, voice))
        with open(filepath, "wb") as audio_file:
            audio_file.write(text.encode("utf-8"))
        active -= 1

    monkeypatch.setattr(audio_service, "_synthesize_chunk", fake_synthesize)
    text = ("தமிழ் வாக்கியம். " * 180)

    res = generate_audio(text, "Tamil", audio_id="test-tamil", is_full_document=True)

    assert res["success"]
    assert generated
    assert all(voice == "ta-IN-PallaviNeural" for _, voice in generated)
    chunks = audio_service._chunk_text(_clean_text_for_speech(text))
    assert Counter(chunk for chunk, _ in generated) == Counter(chunks)
    assert max_active > 1
    assert max_active <= 4
    with open(res["filepath"], "rb") as audio_file:
        assert audio_file.read() == "".join(chunks).encode("utf-8")
    os.remove(res["filepath"])


def test_generate_audio_handles_network_tts_error(monkeypatch):
    async def fake_synthesize(*args, **kwargs):
        raise OSError("Cannot connect to host speech.platform.bing.com:443 ssl: ... getaddrinfo failed")

    monkeypatch.setattr(audio_service, "_synthesize_chunk", fake_synthesize)
    res = generate_audio("Testing audio synthesis.", "English")
    assert not res["success"]
    assert "internet connection" in res["error"].lower()
    assert "speech service" in res["error"].lower()


def test_generate_audio_rejects_unknown_language():
    res = generate_audio("Testing audio synthesis.", "Unknown")
    assert not res["success"]
    assert "not configured" in res["error"]
