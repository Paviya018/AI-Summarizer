from services.document_service import extract_text
from services.text_service import clean_text
from services.chunk_service import split_text


file_path = r"E:\AI-Document-Summarizer\documents\Refined Final year report.pdf"


# Extract
text = extract_text(file_path)


# Clean
cleaned_text = clean_text(text)


# Split into chunks
chunks = split_text(cleaned_text)


print("\n===== CHUNK INFORMATION =====")

print("Total characters:", len(cleaned_text))
print("Total chunks:", len(chunks))


for i, chunk in enumerate(chunks, start=1):

    print(f"\n--- CHUNK {i} ---")
    print("Characters:", len(chunk))
    print(chunk[:300])