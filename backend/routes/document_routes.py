"""
document_routes.py
==================
REST endpoints for uploading, extracting, and summarizing documents.
"""

import logging
from flask import Blueprint, request, jsonify
from pathlib import Path

import config
from utils.validators import validate_file_upload, validate_summarize_request
from utils.file_utils import generate_doc_id, save_uploaded_file, cleanup_old_files
from services.document_service import extract_text
from services.text_service import clean_text
from services.language_service import detect_language
from services.chunk_service import split_text, chunk_summary
from services.document_summarizer import summarize_document

logger = logging.getLogger(__name__)

document_bp = Blueprint("document", __name__)

# In-memory document storage: doc_id -> metadata & text
DOC_CACHE: dict[str, dict] = {}


@document_bp.route("/upload", methods=["POST"])
def upload_document():
    """
    POST /api/documents/upload
    Receives an uploaded file, extracts text, detects language, and chunks it.
    Returns doc_id, detected_language, character count, and text preview.
    """
    cleanup_old_files()

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file field in request."}), 400

    file = request.files["file"]
    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    ocr_language = request.form.get("ocr_language", "en").strip().lower()
    if ocr_language not in config.OCR_LANGUAGE_OPTIONS:
        return jsonify({
            "success": False,
            "error": "Unsupported OCR language. Choose one of: "
            + ", ".join(config.OCR_LANGUAGE_OPTIONS.values()) + ".",
        }), 400

    doc_id = generate_doc_id()
    saved_path = save_uploaded_file(file, doc_id)

    try:
        logger.info("Extracting text from: %s", saved_path.name)
        raw_text = extract_text(str(saved_path), ocr_language)

        if not raw_text or not raw_text.strip():
            return jsonify({
                "success": False,
                "error": "Could not extract any readable text from this document. It may be empty or corrupted."
            }), 422

        cleaned = clean_text(raw_text)
        detected_lang = detect_language(cleaned)
        chunks = split_text(cleaned)
        stats = chunk_summary(chunks)

        DOC_CACHE[doc_id] = {
            "filename": file.filename,
            "raw_text": cleaned,
            "detected_language": detected_lang,
            "chunks": chunks,
            "saved_path": str(saved_path),
        }

        # Generate a safe preview (first 1000 characters)
        preview = cleaned[:1000] + ("..." if len(cleaned) > 1000 else "")

        return jsonify({
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "detected_language": detected_lang,
            "character_count": len(cleaned),
            "chunk_count": len(chunks),
            "chunks_stats": stats,
            "preview": preview,
        }), 200

    except Exception as e:
        logger.exception("Failed to process document upload")
        return jsonify({
            "success": False,
            "error": f"Failed to process document: {str(e)}"
        }), 500


@document_bp.route("/summarize", methods=["POST"])
def summarize_doc():
    """
    POST /api/documents/summarize
    Accepts:
      - doc_id: str (from previous upload) OR text: str
      - summary_length: 'short' | 'medium' | 'detailed' (default: 'medium')
      - target_language: output language string (default: 'English')
    """
    data = request.get_json(silent=True) or {}
    is_valid, error_msg = validate_summarize_request(data)
    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    summary_length = data.get("summary_length", "medium")
    target_language = data.get("target_language", "English")

    doc_id = data.get("doc_id")
    if doc_id:
        if doc_id not in DOC_CACHE:
            return jsonify({
                "success": False,
                "error": "Document session expired or not found. Please re-upload your document."
            }), 404

        doc_data = DOC_CACHE[doc_id]
        chunks = doc_data["chunks"]
        source_language = doc_data["detected_language"]
    else:
        text = clean_text(data.get("text", ""))
        chunks = split_text(text)
        source_language = detect_language(text)

    try:
        logger.info(
            "Summarizing: chunks=%d, length=%s, target_lang=%s",
            len(chunks), summary_length, target_language
        )
        result = summarize_document(
            chunks=chunks,
            summary_length=summary_length,
            target_language=target_language,
            clear_cache=True,
        )

        if result.get("errors") and not result.get("summary"):
            return jsonify({
                "success": False,
                "error": "Summarization failed: " + "; ".join(result["errors"]),
                "mock": result.get("mock", False)
            }), 502

        # Cache final summary for audio generation
        if doc_id and doc_id in DOC_CACHE:
            DOC_CACHE[doc_id]["final_summary"] = result["summary"]
            DOC_CACHE[doc_id]["summary_language"] = target_language

        return jsonify({
            "success": True,
            "summary": result["summary"],
            "source_language": source_language,
            "target_language": target_language,
            "chunk_count": result["chunk_count"],
            "mock": result["mock"],
            "errors": result.get("errors", []),
        }), 200

    except Exception as e:
        logger.exception("Summarization endpoint error")
        return jsonify({
            "success": False,
            "error": f"Internal summarization error: {str(e)}"
        }), 500


@document_bp.route("/audio", methods=["POST"])
def create_audio():
    """
    POST /api/documents/audio
    Converts summary text into speech.
    Accepts:
      - doc_id: str (uses cached final_summary and target_language)
      OR
      - text: str, language: str (default 'English')
    """
    from services.audio_service import generate_audio
    from flask import request, url_for

    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id")

    if doc_id and doc_id in DOC_CACHE and "final_summary" in DOC_CACHE[doc_id]:
        text = DOC_CACHE[doc_id]["final_summary"]
        language = DOC_CACHE[doc_id].get("summary_language", "English")
    else:
        text = data.get("text", "")
        language = data.get("language", "English")

    if not text or not text.strip():
        return jsonify({
            "success": False,
            "error": "No text provided or document summary has not been generated yet."
        }), 400

    result = generate_audio(text=text, language=language, audio_id=doc_id)
    if not result["success"]:
        return jsonify(result), 500

    audio_url = f"/api/documents/audio/{result['filename']}"

    return jsonify({
        "success": True,
        "filename": result["filename"],
        "audio_url": audio_url,
        "language": result["language"],
    }), 200


@document_bp.route("/full-audio", methods=["POST"])
def create_full_document_audio():
    """
    POST /api/documents/full-audio
    Converts the entire document directly into speech without summarizing.
    If target_language is provided and different from the source document language,
    translates the full text before synthesizing.
    """
    from services.audio_service import generate_audio
    from services.translation_service import translate_text

    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id")
    target_language = data.get("target_language", "Original")

    if not doc_id or doc_id not in DOC_CACHE:
        return jsonify({
            "success": False,
            "error": "Document session expired or not found. Please re-upload your document."
        }), 404

    doc_data = DOC_CACHE[doc_id]
    raw_text = doc_data.get("raw_text", "")
    source_language = doc_data.get("detected_language", "English")

    if not raw_text or not raw_text.strip():
        return jsonify({
            "success": False,
            "error": "The document contains no readable text."
        }), 400

    if target_language != "Original" and target_language not in config.SUPPORTED_OUTPUT_LANGUAGES:
        return jsonify({
            "success": False,
            "error": f"Unsupported target language '{target_language}'.",
        }), 400

    # Determine if translation is requested
    final_text = raw_text
    needs_translation = (
        target_language != "Original" and
        target_language.lower() != source_language.lower()
    )

    if needs_translation:
        logger.info("Translating full document (%d chars) from %s to %s...",
                    len(raw_text), source_language, target_language)
        try:
            final_text = translate_text(
                raw_text,
                target_language=target_language,
                source_language=source_language,
            )
        except RuntimeError as e:
            return jsonify({
                "success": False,
                "error": str(e),
            }), 502

    audio_lang = source_language if target_language == "Original" else target_language

    logger.info("Generating full document audio for %s (chars: %d)...", audio_lang, len(final_text))
    result = generate_audio(
        text=final_text,
        language=audio_lang,
        audio_id=f"full_{doc_id}",
        is_full_document=True,
    )

    if not result.get("success"):
        return jsonify(result), 500

    audio_url = f"/api/documents/audio/{result['filename']}"

    # Return preview and metadata along with the audio file
    return jsonify({
        "success": True,
        "filename": result["filename"],
        "audio_url": audio_url,
        "language": audio_lang,
        "character_count": len(final_text),
        "document_text_preview": final_text[:2000] + ("..." if len(final_text) > 2000 else ""),
        "full_text": final_text,
    }), 200


@document_bp.route("/audio/<filename>", methods=["GET"])
def stream_audio(filename):
    """
    GET /api/documents/audio/<filename>
    Streams or downloads the generated MP3 audio file.
    """
    from flask import send_from_directory, abort
    from werkzeug.utils import secure_filename

    safe_name = secure_filename(filename)
    audio_file = config.AUDIO_DIR / safe_name

    if not audio_file.exists():
        abort(404, description="Audio file not found or expired.")

    return send_from_directory(
        directory=str(config.AUDIO_DIR),
        path=safe_name,
        mimetype="audio/mpeg",
        as_attachment=request.args.get("download", "false").lower() == "true",
        download_name=safe_name,
    )
