from types import SimpleNamespace

from services import ocr_service


class FakeOCR:
    def predict(self, file_path):
        return [
            SimpleNamespace(
                json={
                    "res": {
                        "rec_texts": ["First line", "", "Second line"],
                    }
                }
            )
        ]


def test_extract_text_from_image_joins_non_empty_recognized_lines(monkeypatch):
    monkeypatch.setattr(ocr_service, "_get_ocr", lambda language: FakeOCR())

    assert ocr_service.extract_text_from_image("sample.png") == (
        "First line\nSecond line"
    )


def test_extract_text_from_image_automatically_uses_highest_confidence(monkeypatch):
    class LanguageOCR:
        def __init__(self, language):
            self.language = language

        def predict(self, file_path):
            confidence = 0.95 if self.language == "el" else 0.5
            text = "Ελληνικά" if self.language == "el" else "recognized"
            return [
                SimpleNamespace(
                    json={
                        "res": {
                            "rec_texts": [text],
                            "rec_scores": [confidence],
                        }
                    }
                )
            ]

    monkeypatch.setattr(
        ocr_service,
        "_get_ocr",
        lambda language: LanguageOCR(language),
    )

    assert ocr_service.extract_text_from_image("sample.png") == "Ελληνικά"


def test_extract_text_from_image_rejects_unsupported_language():
    try:
        ocr_service.extract_text_from_image("sample.png", "unknown")
    except ValueError as error:
        assert "Unsupported OCR language" in str(error)
    else:
        raise AssertionError("Unsupported OCR languages must be rejected.")
