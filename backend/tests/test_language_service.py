"""
Unit tests for language_service.py
"""

from services.language_service import detect_language


def test_detect_empty_or_whitespace():
    assert detect_language("") == "Unknown"
    assert detect_language("   ") == "Unknown"


def test_detect_english():
    text = "Artificial intelligence and machine learning are revolutionizing software development."
    assert detect_language(text) == "ENGLISH"


def test_detect_greek():
    text = "Ο Θεός και ο άνθρωπος διαβάζουν το βιβλίο."
    assert detect_language(text) == "GREEK"


def test_detect_tamil():
    text = "செயற்கை நுண்ணறிவு பல துறைகளில் பெரும் மாற்றங்களை ஏற்படுத்தி வருகிறது."
    assert detect_language(text) == "TAMIL"


def test_detect_hindi():
    text = "कृत्रिम बुद्धिमत्ता आधुनिक तकनीक का एक महत्वपूर्ण हिस्सा है।"
    assert detect_language(text) == "HINDI"


def test_detect_kannada():
    assert detect_language("ಕನ್ನಡ ಭಾಷೆ ಒಂದು ಸುಂದರ ಭಾಷೆಯಾಗಿದೆ.") == "KANNADA"


def test_detect_malayalam():
    assert detect_language("മലയാളം മനോഹരമായ ഒരു ഭാഷയാണ്.") == "MALAYALAM"
