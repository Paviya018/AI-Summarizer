"""
validators.py
=============
Validates incoming file uploads and API request parameters.
"""

import os
from pathlib import Path
from werkzeug.datastructures import FileStorage
import config


def validate_file_upload(file: FileStorage) -> tuple[bool, str]:
    """
    Validates an uploaded file.
    Checks:
    - File presence
    - Non-empty filename
    - Allowed extensions (.pdf, .docx, .txt, .jpg, .png)
    - File size within limit (MAX_UPLOAD_SIZE_BYTES)
    """
    if not file or not file.filename:
        return False, "No file provided or filename is empty."

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        return False, f"Unsupported file extension '{ext}'. Allowed extensions: {allowed}"

    # Check file size by seeking to end and back
    file.seek(0, os.SEEK_END)
    size_bytes = file.tell()
    file.seek(0)

    if size_bytes == 0:
        return False, "Uploaded file is empty (0 bytes)."

    if size_bytes > config.MAX_UPLOAD_SIZE_BYTES:
        return (
            False,
            f"File size exceeds maximum allowed limit of {config.MAX_UPLOAD_SIZE_MB}MB.",
        )

    return True, ""


def validate_summarize_request(data: dict) -> tuple[bool, str]:
    """
    Validates request payload for /api/documents/summarize.
    """
    if not isinstance(data, dict):
        return False, "Request body must be a JSON object."

    text = data.get("text", "")
    doc_id = data.get("doc_id", "")

    if not text and not doc_id:
        return False, "Either 'text' or 'doc_id' must be provided."

    length = data.get("summary_length", "medium")
    if length not in config.SUMMARY_LENGTH_INSTRUCTIONS:
        valid_lengths = ", ".join(config.SUMMARY_LENGTH_INSTRUCTIONS.keys())
        return False, f"Invalid summary_length '{length}'. Valid choices: {valid_lengths}"

    return True, ""
