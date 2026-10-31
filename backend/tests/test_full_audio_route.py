from app import app
from routes import document_routes
from services import audio_service, translation_service


def test_health_reports_separate_translation_provider():
    with app.test_client() as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.get_json()["translation_provider"] == "Gemini"


def test_health_reports_nllb_translation_provider(monkeypatch):
    import config

    monkeypatch.setattr(config, "TRANSLATION_PROVIDER", "nllb")
    with app.test_client() as client:
        response = client.get("/api/health")
    assert response.get_json()["translation_provider"] == "NLLB"


def test_full_audio_translates_and_synthesizes_in_selected_language(monkeypatch):
    doc_id = "full-audio-language-test"
    monkeypatch.setitem(
        document_routes.DOC_CACHE,
        doc_id,
        {
            "raw_text": "Hello from the original document.",
            "detected_language": "ENGLISH",
        },
    )
    translation_calls = []
    audio_calls = []

    def fake_translate(text, target_language, source_language):
        translation_calls.append((text, target_language, source_language))
        return "தமிழில் மொழிபெயர்க்கப்பட்ட உரை."

    def fake_generate_audio(text, language, audio_id, is_full_document):
        audio_calls.append((text, language, is_full_document))
        return {
            "success": True,
            "filename": "full_audio_test.mp3",
        }

    monkeypatch.setattr(translation_service, "translate_text", fake_translate)
    monkeypatch.setattr(audio_service, "generate_audio", fake_generate_audio)

    with app.test_client() as client:
        response = client.post(
            "/api/documents/full-audio",
            json={"doc_id": doc_id, "target_language": "Tamil"},
        )

    assert response.status_code == 200
    assert translation_calls == [
        ("Hello from the original document.", "Tamil", "ENGLISH")
    ]
    assert audio_calls == [("தமிழில் மொழிபெயர்க்கப்பட்ட உரை.", "Tamil", True)]
    assert response.get_json()["language"] == "Tamil"


def test_full_audio_does_not_synthesize_when_translation_is_incomplete(monkeypatch):
    doc_id = "full-audio-incomplete-translation-test"
    monkeypatch.setitem(
        document_routes.DOC_CACHE,
        doc_id,
        {
            "raw_text": "Hello from the original document.",
            "detected_language": "ENGLISH",
        },
    )
    monkeypatch.setattr(
        translation_service,
        "translate_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("AI translation failed: incomplete translation.")
        ),
    )
    monkeypatch.setattr(
        audio_service,
        "generate_audio",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Audio must not be generated from untranslated text.")
        ),
    )

    with app.test_client() as client:
        response = client.post(
            "/api/documents/full-audio",
            json={"doc_id": doc_id, "target_language": "Tamil"},
        )

    assert response.status_code == 502
    assert "incomplete translation" in response.get_json()["error"]
