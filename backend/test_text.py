from services.document_service import extract_text
from services.text_service import clean_text


file_path = r"E:\AI-Document-Summarizer\documents\Refined Final year report.pdf"

text = extract_text(file_path)

cleaned_text = clean_text(text)

print("\n===== ORIGINAL =====\n")
print(text[:2000])

print("\n\n===== CLEANED =====\n")
print(cleaned_text[:2000])