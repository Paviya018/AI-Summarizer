"""
file_utils.py
=============
Safe file operations and temporary session file management.
"""

import os
import uuid
import time
from pathlib import Path
from werkzeug.utils import secure_filename
import config


def generate_doc_id() -> str:
    """Generate a unique document identifier."""
    return uuid.uuid4().hex[:12]


def save_uploaded_file(file_storage, doc_id: str) -> Path:
    """
    Saves an uploaded file safely to UPLOADS_DIR using doc_id prefix.
    Returns the absolute Path of the saved file.
    """
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    clean_name = secure_filename(file_storage.filename)
    safe_name = f"{doc_id}_{clean_name}"
    target_path = config.UPLOADS_DIR / safe_name
    file_storage.save(str(target_path))
    return target_path


def cleanup_file(path: Path) -> None:
    """Safely removes a file if it exists."""
    try:
        if path and path.exists():
            path.unlink()
    except OSError:
        pass


def cleanup_old_files(max_age_seconds: int = 3600) -> None:
    """
    Clean up files older than max_age_seconds in uploads and audio directories.
    """
    now = time.time()
    for directory in [config.UPLOADS_DIR, config.AUDIO_DIR]:
        if not directory.exists():
            continue
        for file_path in directory.iterdir():
            if file_path.is_file():
                try:
                    if now - file_path.stat().st_mtime > max_age_seconds:
                        file_path.unlink()
                except OSError:
                    pass
