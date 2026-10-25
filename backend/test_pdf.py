from services.pdf_service import extract_text_from_pdf

pdf_path = r"E:\AI-Document-Summarizer\documents\Refined Final year report.pdf"

text = extract_text_from_pdf(pdf_path)

print(text)