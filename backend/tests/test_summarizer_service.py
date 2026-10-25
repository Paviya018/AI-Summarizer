from types import SimpleNamespace

from google.genai import errors as genai_errors

from services import summarizer_service


def test_summarize_retries_temporary_service_unavailable(monkeypatch):
    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, model, contents):
            self.calls += 1
            if self.calls < 3:
                raise genai_errors.APIError(
                    503,
                    {"error": {"message": "temporarily unavailable", "status": "UNAVAILABLE"}},
                )
            return SimpleNamespace(text="A successful summary.")

    models = FakeModels()
    monkeypatch.setattr(
        summarizer_service,
        "_get_client",
        lambda: SimpleNamespace(models=models),
    )
    monkeypatch.setattr(summarizer_service.config, "USE_MOCK_SUMMARIZER", False)
    monkeypatch.setattr(summarizer_service.config, "AI_PROVIDER", "gemini")
    monkeypatch.setattr(summarizer_service.config, "GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(summarizer_service.time, "sleep", lambda _: None)

    result = summarizer_service.summarize_text("A sufficiently short source document.")

    assert result == {
        "summary": "A successful summary.",
        "mock": False,
        "error": "",
    }
    assert models.calls == 3


def test_summarize_falls_back_to_ollama_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(summarizer_service.config, "USE_MOCK_SUMMARIZER", False)
    monkeypatch.setattr(summarizer_service.config, "AI_PROVIDER", "gemini")
    monkeypatch.setattr(summarizer_service.config, "GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(
        summarizer_service,
        "_get_client",
        lambda: (_ for _ in ()).throw(RuntimeError("Gemini unavailable")),
    )
    monkeypatch.setattr(
        summarizer_service,
        "_generate_ollama_content",
        lambda prompt: SimpleNamespace(text="Ollama summary."),
    )

    result = summarizer_service.summarize_text("Source document.")

    assert result == {"summary": "Ollama summary.", "mock": False, "error": ""}


def test_summarize_uses_ollama_when_gemini_key_is_missing(monkeypatch):
    monkeypatch.setattr(summarizer_service.config, "USE_MOCK_SUMMARIZER", False)
    monkeypatch.setattr(summarizer_service.config, "AI_PROVIDER", "gemini")
    monkeypatch.setattr(summarizer_service.config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(
        summarizer_service,
        "_generate_ollama_content",
        lambda prompt: SimpleNamespace(text="Ollama summary."),
    )

    result = summarizer_service.summarize_text("Source document.")

    assert result == {"summary": "Ollama summary.", "mock": False, "error": ""}


def test_ollama_mode_does_not_fall_back_to_cloud(monkeypatch):
    monkeypatch.setattr(summarizer_service.config, "AI_PROVIDER", "ollama")
    monkeypatch.setattr(summarizer_service.config, "GEMINI_API_KEY", "test-gemini-key")

    assert summarizer_service._configured_providers() == ["ollama"]


def test_ollama_request_uses_local_generate_api(monkeypatch):
    captured = {}

    class FakeResponse:
        ok = True

        def json(self):
            return {"response": "Local model response."}

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(summarizer_service.requests, "post", fake_post)
    monkeypatch.setattr(
        summarizer_service.config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
    )
    monkeypatch.setattr(summarizer_service.config, "OLLAMA_MODEL", "qwen3:1.7b")

    result = summarizer_service._generate_ollama_content("Test prompt.")

    assert result.text == "Local model response."
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["json"]["model"] == "qwen3:1.7b"
    assert captured["json"]["prompt"] == "/no_think\n\nTest prompt."
    assert captured["json"]["stream"] is False
    assert captured["json"]["think"] is False
    assert captured["json"]["options"] == {
        "num_ctx": 4096,
        "num_predict": 1536,
        "temperature": 0.2,
        "repeat_penalty": 1.2,
        "repeat_last_n": 128,
    }


