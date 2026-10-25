# AI-Powered Multilingual Document Summarizer with Audio Output

An end-to-end AI document intelligence platform that extracts text from digital and scanned documents (PDF, DOCX, TXT, images), detects language, creates concise summaries with Gemini and Ollama fallback, translates across multiple languages, and synthesizes summaries into natural speech.

---

## 🚀 Key Features

- **Multi-Format Document Ingestion**:
  - Digital PDFs extracted with **PyMuPDF (`fitz`)**.
  - Scanned PDFs and image-based documents extracted with **PaddleOCR**.
  - Image and scanned-PDF OCR uses lightweight models; choose the source text language before upload (English, Greek, Tamil, Telugu, Hindi, Kannada, or Latin-script languages).
  - Native support for Word (`.docx`), plain text (`.txt`), and images (`.png`, `.jpg`).
- **Multilingual Language Detection**:
  - Automatically identifies document language using **Lingua** (Tamil, Hindi, Telugu, Kannada, Malayalam, Greek, English, French, German, Spanish).
- **Intelligent Chunking Engine**:
  - Respects paragraph and sentence boundaries (never severs sentences mid-thought).
  - Contextual overlap between chunks.
- **AI-Powered Summarization with local Ollama or cloud provider failover**:
  - Local-only mode uses Ollama and Qwen3 (`qwen3:1.7b`) for summarization and translation without sending document text to cloud providers.
  - Local inference disables Qwen3's extended thinking and limits repetitive/overlong generations. Full-document translation still takes time on CPU-only hardware.
  - Set `AI_PROVIDER=ollama` in `backend/.env`; local mode does not fall back to paid cloud APIs.
  - Gemini is used first (`gemini-3.8-flash`) and Ollama (`qwen3:1.7b`) is used if Gemini fails.
  - Configure the Gemini API key in `backend/.env`, or set `AI_PROVIDER=ollama` for local-only inference.
  - Configurable length: **Short** (2–3 sentences), **Medium** (1 paragraph), or **Detailed**.
  - Cross-language summarization (e.g. upload English document, generate Tamil summary).
  - Summary and full-document translation use the configured providers and fail over between them when a provider fails.
  - **Checkpoint resumption**: Per-chunk checkpointing ensures that interrupted jobs resume without repeating processed sections.
  - **Safe mock fallback**: Built-in mock mode keeps the pipeline testable without consuming API quota.
- **Multilingual Audio Generation (TTS)**:
  - Generates speech with Microsoft Edge neural voices selected for the requested language.
  - In-browser HTML5 audio player and one-click MP3 download.
- **Modern Responsive Web UI**:
  - React + Vite interface with drag-and-drop file upload, extraction preview, interactive progress indicators, and clipboard copy.

---

## 🏗️ Architecture

```
AI-Document-Summarizer/
├── backend/
│   ├── app.py                     # Flask application factory & CORS setup
│   ├── config.py                  # Central configuration & .env validation
│   ├── requirements.txt           # Pinned Python dependencies
│   ├── .env.example               # Environment variables template
│   ├── routes/
│   │   └── document_routes.py     # REST API endpoints (/upload, /summarize, /audio)
│   ├── services/
│   │   ├── pdf_service.py         # PyMuPDF digital PDF text extraction
│   │   ├── ocr_service.py         # PaddleOCR image extraction
│   │   ├── pdf_ocr_service.py     # PDF page rendering to image & OCR fallback
│   │   ├── docx_service.py        # Word document extraction
│   │   ├── txt_service.py         # Plain text file extraction
│   │   ├── document_service.py    # Document routing & digital/scanned auto-detection
│   │   ├── text_service.py        # Text normalization & whitespace cleaning
│   │   ├── language_service.py    # Lingua language detector
│   │   ├── chunk_service.py       # Smart sentence-boundary chunking
│   │   ├── summarizer_service.py  # Gemini/Ollama summarizer failover with mock fallback
│   │   ├── summary_storage.py     # Checkpoint save/load/clear for chunk summaries
│   │   ├── document_summarizer.py # Document summarization orchestrator
│   │   ├── mock_summarizer.py     # Zero-quota development summarizer
│   │   └── audio_service.py       # Microsoft Edge neural text-to-speech audio generator
│   ├── utils/
│   │   ├── validators.py          # File extension and upload size validation
│   │   └── file_utils.py          # Safe filenames and temporary file cleanup
│   ├── tests/
│   │   ├── test_chunk_service.py
│   │   ├── test_language_service.py
│   │   ├── test_audio_service.py
│   │   └── test_document_summarizer.py
│   ├── uploads/                   # Stored uploaded files (.gitkeep)
│   ├── generated_audio/           # Stored generated MP3 files (.gitkeep)
│   └── data/                      # Persistent chunk checkpoints (.gitkeep)
└── frontend/
    ├── src/
    │   ├── App.jsx                # Main application component
    │   ├── App.css                # Dark mode styling & animations
    │   └── main.jsx               # React DOM entry point
    ├── vite.config.js             # Vite proxy configuration (/api -> :5000)
    └── package.json
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 2. Backend Setup
```powershell
cd backend

# Create & activate virtual environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# For local-only AI, install Ollama, pull qwen3:1.7b, and set AI_PROVIDER=ollama.
# Local mode does not require cloud API keys and never sends text to cloud providers.
# For cloud mode, add one or both provider API keys.
# Keep USE_MOCK_SUMMARIZER=true to avoid external API calls.
```

### 3. Frontend Setup
```powershell
cd frontend
npm install
```

---

## 🚦 Running the Application

### Start Backend API Server
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python app.py
```
> Running at **`http://127.0.0.1:5000`**

### Start Frontend Dev Server
```powershell
cd frontend
npm run dev
```
> Running at **`http://localhost:5173`**

For images and scanned PDFs, select the language used in the source image before uploading. OCR is less reliable for blurry, noisy, rotated, or handwritten text.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check, active model, and supported languages |
| `POST` | `/api/documents/upload` | Upload document, extract text, detect language, chunk text |
| `POST` | `/api/documents/summarize` | Generate AI summary with specified length and target language |
| `POST` | `/api/documents/audio` | Synthesize speech for summary in target language |
| `GET` | `/api/documents/audio/<filename>` | Stream or download generated MP3 audio file |

---

## 🧪 Running Automated Tests

Run the full pytest suite from the `backend/` directory:
```powershell
pytest tests/ -v
```
All unit and integration tests run in mock mode by default to prevent consuming API quota during testing.
