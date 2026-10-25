import fitz
import os
from services.ocr_service import extract_text_from_image


def extract_text_from_scanned_pdf(file_path, ocr_language="en"):

    document = fitz.open(file_path)

    extracted_text = []

    for page_number, page in enumerate(document):

        print(f"OCR processing page {page_number + 1}...")

        # Render PDF page as image
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2)
        )

        image_path = f"temp_page_{page_number + 1}.png"

        pix.save(image_path)

        # OCR the image
        text = extract_text_from_image(image_path, ocr_language)

        if text.strip():
            extracted_text.append(text)

        # Delete temporary image
        os.remove(image_path)

    document.close()

    return "\n\n".join(extracted_text)