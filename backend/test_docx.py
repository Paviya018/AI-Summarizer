from services.docx_service import extract_text_from_docx


docx_path = r"E:\AI-Document-Summarizer\documents\Final year project report paviya.docx"

text = extract_text_from_docx(docx_path)

print(text)