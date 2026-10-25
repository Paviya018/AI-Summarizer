from lingua import Language, LanguageDetectorBuilder


languages = [
    Language.ENGLISH,
    Language.GREEK,
    Language.TAMIL,
    Language.HINDI,
    Language.TELUGU,
    Language.FRENCH,
    Language.GERMAN,
    Language.SPANISH,
]


detector = LanguageDetectorBuilder.from_languages(
    *languages
).build()


def detect_language(text):

    if not text or not text.strip():
        return "Unknown"

    if any("\u0c80" <= char <= "\u0cff" for char in text):
        return "KANNADA"

    if any("\u0d00" <= char <= "\u0d7f" for char in text):
        return "MALAYALAM"

    language = detector.detect_language_of(text)

    if language is None:
        return "Unknown"

    return language.name