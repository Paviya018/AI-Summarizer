from app import app
from routes import document_routes
from services import audio_service


def test_translated_summary_is_the_text_sent_to_tamil_voice(monkeypatch):
    doc_id = "translated-summary-audio-test"
    monkeypatch.setitem(
        document_routes.DOC_CACHE,
        doc_id,
        {
            "chunks": ["A source document in English."],
            "detected_language": "ENGLISH",
        },
    )
    monkeypatch.setattr(
        document_routes,
        "summarize_document",
        lambda **kwargs: {
            "summary": "தமிழ் சுருக்கம்.",
            "chunk_count": 1,
            "mock": False,
            "errors": [],
        },
    )
    audio_calls = []

    def fake_generate_audio(text, language, audio_id):
        audio_calls.append((text, language, audio_id))
        return {
            "success": True,
            "filename": "summary_audio_test.mp3",
            "language": language,
        }

    monkeypatch.setattr(audio_service, "generate_audio", fake_generate_audio)

    with app.test_client() as client:
        summary_response = client.post(
            "/api/documents/summarize",
            json={
                "doc_id": doc_id,
                "summary_length": "short",
                "target_language": "Tamil",
            },
        )
        audio_response = client.post(
            "/api/documents/audio",
            json={"doc_id": doc_id},
        )

    assert summary_response.status_code == 200
    assert summary_response.get_json()["summary"] == "தமிழ் சுருக்கம்."
    assert audio_response.status_code == 200
    assert audio_calls == [("தமிழ் சுருக்கம்.", "Tamil", doc_id)]
