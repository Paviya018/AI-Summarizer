import sys
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from services import translation_service
from services.translation_service import translate_text


def test_translation_uses_gemini_with_requested_languages(monkeypatch):
    calls = []
    translated_text = "செயற்கை நுண்ணறிவு கணினி அறிவியலின் ஒரு பிரிவாகும்."

    def fake_generate(prompt):
        calls.append(prompt)
        return type("Response", (), {"text": translated_text})()

    monkeypatch.setattr(translation_service, "generate_gemini_content", fake_generate)

    translated = translate_text(
        "Artificial intelligence is a branch of computer science.",
        "Tamil",
        "ENGLISH",
    )

    assert translated == translated_text
    assert len(calls) == 1
    assert "Translate from ENGLISH." in calls[0]
    assert "into Tamil" in calls[0]
    assert "Artificial intelligence is a branch of computer science." in calls[0]


def test_nllb_translation_uses_language_codes(monkeypatch):
    calls = []

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        model_max_length = 1024
        unk_token_id = 2

        def convert_tokens_to_ids(self, token):
            return {"eng_Latn": 0, "tam_Taml": 1}.get(token, self.unk_token_id)

        def __call__(self, text, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                return FakeInputs(input_ids=[text])
            return {"input_ids": list(text)}

        def batch_decode(self, output, skip_special_tokens):
            return ["translated"]

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return ["output"]

    class FakeTorch:
        @staticmethod
        def inference_mode():
            from contextlib import nullcontext

            return nullcontext()

    tokenizer = FakeTokenizer()
    model = FakeModel()
    monkeypatch.setattr(
        translation_service,
        "_nllb_components",
        (tokenizer, model, "cpu", FakeTorch),
    )
    monkeypatch.setattr(translation_service.config, "TRANSLATION_PROVIDER", "nllb")
    monkeypatch.setattr(
        translation_service,
        "generate_gemini_content",
        lambda prompt: (_ for _ in ()).throw(AssertionError("Gemini must not run")),
    )

    assert translate_text("Hello", "Tamil", "English") == "translated"
    assert translate_text("Hello again", "Tamil", "English") == "translated"
    assert len(calls) == 2
    assert all(call["forced_bos_token_id"] == 1 for call in calls)


def test_nllb_model_is_loaded_once(monkeypatch):
    loads = []
    fake_torch = SimpleNamespace(
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_tokenizer = object()
    fake_model = SimpleNamespace(to=lambda device: None, eval=lambda: None)

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_name):
            loads.append(model_name)
            return fake_tokenizer if len(loads) == 1 else fake_model

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeLoader,
            AutoModelForSeq2SeqLM=FakeLoader,
        ),
    )
    monkeypatch.setattr(translation_service, "_nllb_components", None)

    first = translation_service._load_nllb_components()
    second = translation_service._load_nllb_components()

    assert first is second
    assert loads == [
        translation_service.config.NLLB_MODEL,
        translation_service.config.NLLB_MODEL,
    ]


def test_nllb_chunks_text_to_fit_model_token_limit():
    class FakeTokenizer:
        model_max_length = 8

        def __call__(self, text, add_special_tokens):
            return {"input_ids": list(text)}

    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = translation_service._split_nllb_text(text, FakeTokenizer())

    assert "".join(chunks) == text
    assert all(len(chunk) <= 8 for chunk in chunks)


def test_nllb_auto_detects_source_language(monkeypatch):
    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        model_max_length = 1024
        unk_token_id = 2
        src_lang = None

        def convert_tokens_to_ids(self, token):
            return {"eng_Latn": 0, "tam_Taml": 1}.get(token, self.unk_token_id)

        def __call__(self, text, **kwargs):
            if kwargs.get("return_tensors"):
                return FakeInputs(input_ids=[text])
            return {"input_ids": list(text)}

        def batch_decode(self, output, skip_special_tokens):
            return ["translated"]

    class FakeTorch:
        @staticmethod
        def inference_mode():
            from contextlib import nullcontext

            return nullcontext()

    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        translation_service,
        "_nllb_components",
        (
            tokenizer,
            type("Model", (), {"generate": lambda self, **kwargs: ["output"]})(),
            "cpu",
            FakeTorch,
        ),
    )
    monkeypatch.setattr(translation_service.config, "TRANSLATION_PROVIDER", "nllb")
    monkeypatch.setattr(
        translation_service, "_detect_nllb_source_language", lambda text: "ENGLISH"
    )

    assert translate_text("Hello", "Tamil", "auto") == "translated"
    assert tokenizer.src_lang == "eng_Latn"


def test_nllb_rejects_unsupported_source_language(monkeypatch):
    monkeypatch.setattr(translation_service.config, "TRANSLATION_PROVIDER", "nllb")
    with pytest.raises(RuntimeError, match="NLLB does not support language"):
        translate_text("Hello", "Tamil", "Klingon")


def test_translation_supports_all_languages_offered_by_app(monkeypatch):
    import config

    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return type("Response", (), {"text": "translated"})()

    monkeypatch.setattr(translation_service, "generate_gemini_content", fake_generate)

    for language in config.SUPPORTED_OUTPUT_LANGUAGES:
        assert translate_text("Hello", language, "auto") == "translated"

    assert all(
        f"into {language}" in prompt
        for language, prompt in zip(config.SUPPORTED_OUTPUT_LANGUAGES, calls)
    )
    assert all("Detect the source language automatically." in prompt for prompt in calls)


def test_translation_splits_long_text_to_gemini_chunk_limit(monkeypatch):
    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return type("Response", (), {"text": "மொழிபெயர்ப்பு"})()

    monkeypatch.setattr(translation_service, "generate_gemini_content", fake_generate)
    text = "word " * 2000

    assert translate_text(text, "Tamil", "English") == "\n".join(
        "மொழிபெயர்ப்பு" for _ in translation_service._split_for_translation(
            text, max_chars=translation_service.GEMINI_TRANSLATION_CHUNK_CHARS
        )
    )
    assert len(prompts) > 1
    assert all(
        len(chunk) <= translation_service.GEMINI_TRANSLATION_CHUNK_CHARS
        for chunk in translation_service._split_for_translation(
            text, max_chars=translation_service.GEMINI_TRANSLATION_CHUNK_CHARS
        )
    )


def test_translation_rejects_unsupported_target_language():
    with pytest.raises(ValueError, match="Unsupported target language"):
        translate_text("Hello", "Klingon", "English")


def test_translation_does_not_call_provider_for_same_source_and_target(monkeypatch):
    monkeypatch.setattr(
        translation_service,
        "generate_gemini_content",
        lambda prompt: (_ for _ in ()).throw(
            AssertionError("Gemini should not be called")
        ),
    )

    assert translate_text("Hello", "English", "ENGLISH") == "Hello"


def test_translation_reports_provider_errors(monkeypatch):
    monkeypatch.setattr(
        translation_service,
        "generate_gemini_content",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("Gemini is unavailable.")),
    )

    with pytest.raises(RuntimeError, match="Gemini translation failed"):
        translate_text("Hello", "Tamil", "English")


def test_translation_reports_quota_reset_time(monkeypatch):
    quota_error = genai_errors.APIError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Quota exceeded.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "52520s",
                    }
                ],
            }
        },
    )
    monkeypatch.setattr(
        translation_service,
        "generate_gemini_content",
        lambda prompt: (_ for _ in ()).throw(quota_error),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini quota limit exceeded. Please try again after 14 hours, 35 minutes.",
    ):
        translate_text("Hello", "Tamil", "English")


def test_translation_reports_quota_without_retry_time(monkeypatch):
    quota_error = genai_errors.APIError(
        429,
        {"error": {"code": 429, "message": "Quota exceeded.", "status": "RESOURCE_EXHAUSTED"}},
    )
    monkeypatch.setattr(
        translation_service,
        "generate_gemini_content",
        lambda prompt: (_ for _ in ()).throw(quota_error),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini quota limit exceeded. Please try again after the quota resets.",
    ):
        translate_text("Hello", "Tamil", "English")


def test_empty_translation_returns_empty_without_calling_provider(monkeypatch):
    monkeypatch.setattr(
        translation_service,
        "generate_gemini_content",
        lambda prompt: (_ for _ in ()).throw(
            AssertionError("Provider should not be called")
        ),
    )

    assert translate_text("  ", "Tamil", "English") == ""


def test_translation_reports_empty_gemini_response(monkeypatch):
    monkeypatch.setattr(
        translation_service,
        "generate_gemini_content",
        lambda prompt: type("Response", (), {"text": ""})(),
    )

    with pytest.raises(RuntimeError, match="Gemini returned an empty translation"):
        translate_text("Hello", "Tamil", "English")
