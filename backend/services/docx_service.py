from docx import Document


def extract_text_from_docx(file_path):
    document = Document(file_path)

    extracted_text = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            extracted_text.append(paragraph.text)

    return "\n".join(extracted_text)