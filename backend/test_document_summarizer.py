from services.document_service import extract_text
from services.text_service import clean_text
from services.chunk_service import split_text
from services.document_summarizer import summarize_document


file_path = r"E:\AI-Document-Summarizer\documents\Refined Final year report.pdf"


print("Extracting document...")

text = extract_text(file_path)


print("Cleaning text...")

cleaned_text = clean_text(text)


print("Creating chunks...")

chunks = split_text(cleaned_text)


print("\n===== DOCUMENT INFORMATION =====")

print("Total characters:", len(cleaned_text))
print("Total chunks:", len(chunks))


print("\n===== STARTING AI SUMMARIZATION =====")

final_summary = summarize_document(chunks)


print("\n\n===== FINAL DOCUMENT SUMMARY =====")

print(final_summary)