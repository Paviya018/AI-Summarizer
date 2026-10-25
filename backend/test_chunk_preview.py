from services.document_service import extract_text
from services.text_service import clean_text
from services.chunk_service import split_text


file_path = r"E:\AI-Document-Summarizer\documents\Refined Final year report.pdf"

text = extract_text(file_path)

cleaned_text = clean_text(text)

chunks = split_text(cleaned_text)


print("Total characters:", len(cleaned_text))
print("Total chunks:", len(chunks))

print("\n===== FIRST 2 CHUNKS =====")

for i, chunk in enumerate(chunks[:2], start=1):

    print(f"\n--- CHUNK {i} ---")
    print(chunk)