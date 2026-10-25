from threading import Lock

import config
from paddleocr import PaddleOCR


_OCR_MODELS = {
    "en": ("PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"),
    "latin": ("PP-OCRv5_mobile_det", "latin_PP-OCRv5_mobile_rec"),
    "el": ("PP-OCRv5_mobile_det", "el_PP-OCRv5_mobile_rec"),
    "ta": ("PP-OCRv5_mobile_det", "ta_PP-OCRv5_mobile_rec"),
    "te": ("PP-OCRv5_mobile_det", "te_PP-OCRv5_mobile_rec"),
    "devanagari": ("PP-OCRv5_mobile_det", "devanagari_PP-OCRv5_mobile_rec"),
    "ka": ("PP-OCRv3_mobile_det", "ka_PP-OCRv3_mobile_rec"),
}

_ocr = None
_ocr_language = None
_ocr_lock = Lock()


def _get_ocr(language):
    global _ocr, _ocr_language

    if language not in _OCR_MODELS:
        supported = ", ".join(config.OCR_LANGUAGE_OPTIONS.values())
        raise ValueError(f"Unsupported OCR language. Choose one of: {supported}.")

    if _ocr is None or _ocr_language != language:
        detection_model, recognition_model = _OCR_MODELS[language]
        _ocr = PaddleOCR(
            text_detection_model_name=detection_model,
            text_recognition_model_name=recognition_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        _ocr_language = language
    return _ocr


def extract_text_from_image(file_path, language="en"):
    with _ocr_lock:
        result = _get_ocr(language).predict(file_path)

        extracted_text = []

        for res in result:
            data = res.json

            if "res" in data:
                texts = data["res"].get("rec_texts", [])

                for text in texts:
                    if text.strip():
                        extracted_text.append(text)

        return "\n".join(extracted_text)