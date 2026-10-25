from services.document_service import extract_text
from services.text_service import clean_text
from services.chunk_service import split_text
from services.mock_summarizer import mock_summarize


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


print("\n===== TESTING SUMMARIZATION PIPELINE =====")

chunk_summaries = []

for i, chunk in enumerate(chunks, start=1):

    print(f"\nProcessing chunk {i}/{len(chunks)}...")

    summary = mock_summarize(chunk)

    chunk_summaries.append(summary)

    print("Chunk summary:")
    print(summary)


print("\n===== ALL CHUNKS PROCESSED =====")

final_summary = "\n\n".join(chunk_summaries)

print("\n===== FINAL TEST SUMMARY =====")
print(final_summary)