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


def test_extract_text_from_image_rejects_unsupported_language():
    try:
        ocr_service.extract_text_from_image("sample.png", "unknown")
    except ValueError as error:
        assert "Unsupported OCR language" in str(error)
    else:
        raise AssertionError("Unsupported OCR languages must be rejected.")
