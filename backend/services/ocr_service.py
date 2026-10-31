from threading import Lock
import unicodedata

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
_SCRIPT_NAMES = {
    "en": ("LATIN",),
    "latin": ("LATIN",),
    "el": ("GREEK",),
    "ta": ("TAMIL",),
    "te": ("TELUGU",),
    "devanagari": ("DEVANAGARI",),
    "ka": ("KANNADA",),
}


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


def _recognize_text(file_path, language):
    result = _get_ocr(language).predict(file_path)
    extracted_text = []
    scores = []

    for res in result:
        data = res.json
        if "res" not in data:
            continue

        recognition = data["res"]
        for text in recognition.get("rec_texts", []):
            if text.strip():
                extracted_text.append(text)

        scores.extend(recognition.get("rec_scores", []))

    text = "\n".join(extracted_text)
    confidence = sum(scores) / len(scores) if scores else 1
    return text, confidence


def _script_match(text, language):
    letters = [
        unicodedata.name(char, "")
        for char in text
        if char.isalpha()
    ]
    if not letters:
        return 0

    matching_letters = sum(
        any(name.startswith(script) for script in _SCRIPT_NAMES[language])
        for name in letters
    )
    return matching_letters / len(letters)


def extract_text_from_image(file_path, language=None):
    with _ocr_lock:
        if language is not None:
            return _recognize_text(file_path, language)[0]

        best_text = ""
        best_score = -1
        for candidate_language in _OCR_MODELS:
            text, confidence = _recognize_text(file_path, candidate_language)
            score = confidence * (
                0.1 + 0.9 * _script_match(text, candidate_language)
            )
            if text and score > best_score:
                best_text = text
                best_score = score

        return best_text