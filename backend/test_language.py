from services.document_service import extract_text
from services.text_service import clean_text
from services.language_service import detect_language


file_path = r"E:\AI-Document-Summarizer\documents\Refined Final year report.pdf"

text = extract_text(file_path)

cleaned_text = clean_text(text)

language = detect_language(cleaned_text)

print("\n===== LANGUAGE DETECTION =====")
print("Detected language:", language)

