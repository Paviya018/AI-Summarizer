from services.document_service import extract_text


file_path = r"E:\AI-Document-Summarizer\documents\Refined Final year report.pdf"

text = extract_text(file_path)

print("\n===== TOTAL CHARACTERS =====")
print(len(text))

print("\n===== FIRST 3000 CHARACTERS =====")
print(text[:3000])