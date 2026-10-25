import fitz


def extract_text_from_pdf(file_path):

    document = fitz.open(file_path)

    extracted_text = []

    for page in document:
        text = page.get_text().strip()

        if text:
            extracted_text.append(text)

    document.close()

    return "\n".join(extracted_text)