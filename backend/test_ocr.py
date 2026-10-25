from services.ocr_service import extract_text_from_image


image_path = r"E:\AI-Document-Summarizer\documents\wallpaper.jpg"

text = extract_text_from_image(image_path)

print("\n===== EXTRACTED TEXT =====\n")
print(text)