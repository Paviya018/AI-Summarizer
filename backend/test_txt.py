from services.txt_service import extract_text_from_txt


txt_path = r"E:\AI-Document-Summarizer\documents\sample.txt"

text = extract_text_from_txt(txt_path)

print(text)