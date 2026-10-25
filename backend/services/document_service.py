import os

from services.pdf_service import extract_text_from_pdf
from services.pdf_ocr_service import extract_text_from_scanned_pdf
from services.docx_service import extract_text_from_docx
from services.txt_service import extract_text_from_txt
from services.ocr_service import extract_text_from_image


def extract_text(file_path, ocr_language="en"):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":

        # First try normal PDF extraction
        text = extract_text_from_pdf(file_path)

        # Digital PDF
        if len(text.strip()) > 50:
            print("Digital PDF detected.")
            return text

        # Scanned PDF
        print("Scanned PDF detected.")
        print("Starting OCR...")

        return extract_text_from_scanned_pdf(file_path, ocr_language)

    elif extension == ".docx":

        return extract_text_from_docx(file_path)

    elif extension == ".txt":

        return extract_text_from_txt(file_path)

    elif extension in [".jpg", ".jpeg", ".png"]:

        return extract_text_from_image(file_path, ocr_language)

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )