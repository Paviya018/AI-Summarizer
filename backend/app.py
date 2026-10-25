"""
app.py
======
Main entry point for the AI Document Summarizer Flask backend.
"""

import logging
from flask import Flask, jsonify
from flask_cors import CORS

import config
from routes.document_routes import document_bp

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app")


def create_app() -> Flask:
    """Application factory for Flask."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_BYTES

    # Enable CORS for all frontend origins (React default port 5173 / 3000)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register Blueprints
    app.register_blueprint(document_bp, url_prefix="/api/documents")

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "service": "AI Document Summarizer API",
            "status": "running",
            "health_url": "/api/health",
        }), 200

    @app.route("/api/health", methods=["GET"])
    def health():
        warnings = config.validate_config()
        provider_models = {
            "gemini": config.GEMINI_MODEL,
            "ollama": config.OLLAMA_MODEL,
        }
        return jsonify({
            "status": "healthy",
            "model": provider_models.get(config.AI_PROVIDER, config.GEMINI_MODEL),
            "provider": config.AI_PROVIDER,
            "configured_providers": [
                name
                for name, key, placeholder in (
                    ("gemini", config.GEMINI_API_KEY, "your_gemini_api_key_here"),
                )
                if key and key != placeholder
            ] + ["ollama"],
            "mock_mode": config.USE_MOCK_SUMMARIZER,
            "supported_languages": list(config.SUPPORTED_OUTPUT_LANGUAGES.keys()),
            "ocr_languages": config.OCR_LANGUAGE_OPTIONS,
            "summary_lengths": list(config.SUMMARY_LENGTH_INSTRUCTIONS.keys()),
            "warnings": warnings,
        }), 200

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "success": False,
            "error": f"File is too large. Maximum allowed size is {config.MAX_UPLOAD_SIZE_MB}MB."
        }), 413

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "The requested API endpoint does not exist."
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "success": False,
            "error": "Internal server error occurred."
        }), 500

    return app


app = create_app()

if __name__ == "__main__":
    logger.info("Starting AI Document Summarizer Backend on http://127.0.0.1:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=config.FLASK_DEBUG,
        use_reloader=False,
    )
