import React, { useState, useEffect, useRef } from 'react'
import {
  Upload,
  FileText,
  Volume2,
  Download,
  AlertCircle,
  Copy,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Headphones
} from 'lucide-react'
import './App.css'

async function readApiResponse(response) {
  const body = await response.text()
  if (!body.trim()) {
    throw new Error(
      `The server returned an empty response (HTTP ${response.status}). Check that the backend is running and review its terminal logs.`
    )
  }

  try {
    return JSON.parse(body)
  } catch {
    const detail = body.replace(/\s+/g, ' ').slice(0, 200)
    throw new Error(
      `The server returned an invalid response (HTTP ${response.status})${detail ? `: ${detail}` : '.'}`
    )
  }
}

export default function App() {
  // App state
  const [health, setHealth] = useState(null)
  const [ocrLanguage, setOcrLanguage] = useState('en')
  const [isUploading, setIsUploading] = useState(false)
  const [uploadData, setUploadData] = useState(null)
  const [error, setError] = useState('')
  const [showPreview, setShowPreview] = useState(false)

  // Operating Mode: 'summary' or 'full-audio'
  const [activeMode, setActiveMode] = useState('summary')

  // Summarize options
  const [targetLanguage, setTargetLanguage] = useState('English')
  const [summaryLength, setSummaryLength] = useState('medium')
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [summaryData, setSummaryData] = useState(null)
  const [copied, setCopied] = useState(false)

  // Summary Audio generation state
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false)
  const [audioUrl, setAudioUrl] = useState('')
  const [audioFilename, setAudioFilename] = useState('')

  // Full Document Audio state
  const [fullAudioLang, setFullAudioLang] = useState('Original')
  const [isGeneratingFullAudio, setIsGeneratingFullAudio] = useState(false)
  const [fullAudioData, setFullAudioData] = useState(null)
  const [copiedFullText, setCopiedFullText] = useState(false)

  const fileInputRef = useRef(null)

  // Check backend health on mount
  useEffect(() => {
    fetch('/api/health')
      .then(readApiResponse)
      .then((data) => setHealth(data))
      .catch((err) => {
        console.error('Failed to reach backend:', err)
        setError('Cannot connect to backend server. Make sure Flask is running on port 5000.')
      })
  }, [])

  // File drop / select handler
  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return
    setError('')
    setUploadData(null)
    setSummaryData(null)
    setAudioUrl('')
    setFullAudioData(null)

    uploadDocument(selectedFile)
  }

  const uploadDocument = async (docFile) => {
    setIsUploading(true)
    setError('')
    const formData = new FormData()
    formData.append('file', docFile)
    formData.append('ocr_language', ocrLanguage)

    try {
      const res = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      })
      const data = await readApiResponse(res)

      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to upload and extract text.')
      }

      setUploadData(data)
      // Pre-select detected language if recognized
      if (data.detected_language && data.detected_language !== 'Unknown') {
        const titleCaseLang =
          data.detected_language.charAt(0).toUpperCase() +
          data.detected_language.slice(1).toLowerCase()
        if (health?.supported_languages?.includes(titleCaseLang)) {
          setTargetLanguage(titleCaseLang)
        }
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setIsUploading(false)
    }
  }

  // Summarize handler
  const handleSummarize = async () => {
    if (!uploadData?.doc_id) return
    setIsSummarizing(true)
    setError('')
    setAudioUrl('')

    try {
      const res = await fetch('/api/documents/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: uploadData.doc_id,
          summary_length: summaryLength,
          target_language: targetLanguage,
        }),
      })
      const data = await readApiResponse(res)

      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to generate summary.')
      }

      setSummaryData(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsSummarizing(false)
    }
  }

  // Summary Audio generation handler
  const handleGenerateAudio = async () => {
    if (!summaryData?.summary) return
    setIsGeneratingAudio(true)
    setError('')

    try {
      const res = await fetch('/api/documents/audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: uploadData?.doc_id,
          text: summaryData.summary,
          language: targetLanguage,
        }),
      })
      const data = await readApiResponse(res)

      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to generate audio.')
      }

      setAudioUrl(data.audio_url)
      setAudioFilename(data.filename)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsGeneratingAudio(false)
    }
  }

  // Full Document Audio handler (No summary)
  const handleGenerateFullDocumentAudio = async () => {
    if (!uploadData?.doc_id) return
    setIsGeneratingFullAudio(true)
    setError('')
    setFullAudioData(null)

    try {
      const res = await fetch('/api/documents/full-audio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: uploadData.doc_id,
          target_language: fullAudioLang,
        }),
      })
      const data = await readApiResponse(res)

      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Failed to generate full document audio.')
      }

      setFullAudioData(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsGeneratingFullAudio(false)
    }
  }

  const handleCopySummary = () => {
    if (!summaryData?.summary) return
    const clean = summaryData.summary.replace(/^\[MOCK SUMMARY.*?\]\s*/i, '')
    navigator.clipboard.writeText(clean)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleCopyFullText = () => {
    if (!fullAudioData?.full_text) return
    navigator.clipboard.writeText(fullAudioData.full_text)
    setCopiedFullText(true)
    setTimeout(() => setCopiedFullText(false), 2000)
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="hero-layout">
          <div className="hero-copy">
            <span className="eyebrow"><Sparkles size={14} /> Study smarter</span>
            <h1>Turn long notes into <span>clear next steps.</span></h1>
            <p>Upload lecture notes, readings, or research papers. Get a focused summary, translate it, and listen while you review.</p>
          </div>
          <div className="hero-note" aria-hidden="true">
            <div className="hero-note-top"><FileText size={16} /> QUICK REVIEW</div>
            <div className="hero-note-lines"><i /><i /><i /><i /></div>
            <div className="hero-note-sticker"><Headphones size={16} /> Listen anywhere</div>
          </div>
        </div>
        <div className="header-badges">
          {health?.mock_mode && (
            <span className="badge mock">
              ⚠️ Mock Mode Active (Quota Friendly)
            </span>
          )}
        </div>
      </header>

      {/* Global error banner */}
      {error && (
        <div className="alert alert-danger">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* 1. Document Upload Card */}
      <div className="card">
        <div className="section-kicker">01 / Start here</div>
        <h2 className="card-title">
          <Upload size={20} color="var(--accent)" /> 1. Upload Document
        </h2>

        <div className="form-group" style={{ marginBottom: '1rem' }}>
          <label htmlFor="ocr-language">Text language in image or scanned PDF</label>
          <select
            id="ocr-language"
            value={ocrLanguage}
            onChange={(e) => setOcrLanguage(e.target.value)}
            disabled={isUploading}
          >
            {Object.entries(health?.ocr_languages || {
              en: 'English',
              latin: 'French, German, or Spanish (Latin script)',
              el: 'Greek',
              ta: 'Tamil',
              te: 'Telugu',
              devanagari: 'Hindi',
              ka: 'Kannada',
            }).map(([code, language]) => (
              <option key={code} value={code}>{language}</option>
            ))}
          </select>
        </div>

        <div
          className="dropzone"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            if (e.dataTransfer.files?.[0]) {
              handleFileChange(e.dataTransfer.files[0])
            }
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
            style={{ display: 'none' }}
            onChange={(e) => handleFileChange(e.target.files?.[0])}
          />
          <Upload className="dropzone-icon" />
          <h3>Drag and drop your document here, or click to browse</h3>
          <p>Supports Digital & Scanned PDFs (with OCR), DOCX, TXT, PNG, JPG (up to 20MB). Select the image text language before uploading.</p>
        </div>

        {isUploading && (
          <div style={{ textAlign: 'center', marginTop: '1.25rem', color: 'var(--accent)' }}>
            <div className="spinner" style={{ margin: '0 auto 0.5rem' }} />
            <span>Processing and extracting text (PyMuPDF / PaddleOCR)...</span>
          </div>
        )}

        {/* Uploaded Metadata */}
        {uploadData && !isUploading && (
          <div>
            <div className="meta-grid">
              <div className="meta-item">
                <div className="meta-label">File Name</div>
                <div className="meta-value">{uploadData.filename}</div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Detected Language</div>
                <div className="meta-value" style={{ color: '#38bdf8' }}>
                  {uploadData.detected_language}
                </div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Total Characters</div>
                <div className="meta-value">{uploadData.character_count.toLocaleString()}</div>
              </div>
              <div className="meta-item">
                <div className="meta-label">Text Chunks</div>
                <div className="meta-value">{uploadData.chunk_count}</div>
              </div>
            </div>

            {/* Collapsible Text Preview */}
            <div style={{ marginTop: '1rem' }}>
              <button
                className="preview-toggle"
                onClick={() => setShowPreview(!showPreview)}
              >
                <span>Document Text Preview</span>
                {showPreview ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
              {showPreview && (
                <div className="preview-content">
                  {uploadData.preview}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* 2. Mode Selector & Options Card */}
      {uploadData && (
        <div className="card">
          <div className="section-kicker">02 / Choose your study mode</div>
          <div className="mode-tabs">
            <button
              className={`mode-tab ${activeMode === 'summary' ? 'active' : ''}`}
              onClick={() => setActiveMode('summary')}
            >
              <FileText size={18} /> AI Summarize & Speech
            </button>
            <button
              className={`mode-tab ${activeMode === 'full-audio' ? 'active' : ''}`}
              onClick={() => setActiveMode('full-audio')}
            >
              <Headphones size={18} /> Read Full Document (No Summary)
            </button>
          </div>

          {/* TAB 1: AI Summarize Mode */}
          {activeMode === 'summary' && (
            <div>
              <h2 className="card-title">
                <Sparkles size={20} color="var(--accent)" /> Summary Settings
              </h2>

              <div className="controls-grid">
                <div className="form-group">
                  <label>Output Language</label>
                  <select
                    value={targetLanguage}
                    onChange={(e) => setTargetLanguage(e.target.value)}
                  >
                    {health?.supported_languages?.map((lang) => (
                      <option key={lang} value={lang}>
                        {lang}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label>Summary Length</label>
                  <select
                    value={summaryLength}
                    onChange={(e) => setSummaryLength(e.target.value)}
                  >
                    <option value="short">Short (2-3 sentences)</option>
                    <option value="medium">Medium (1 paragraph, 5-7 sentences)</option>
                    <option value="detailed">Detailed (Comprehensive overview)</option>
                  </select>
                </div>
              </div>

              <button
                className="btn btn-primary"
                onClick={handleSummarize}
                disabled={isSummarizing}
              >
                {isSummarizing ? (
                  <>
                    <div className="spinner" /> Generating AI Summary...
                  </>
                ) : (
                  <>
                    <Sparkles size={18} /> Summarize Document
                  </>
                )}
              </button>
            </div>
          )}

          {/* TAB 2: Full Document Audio Mode (No Summary) */}
          {activeMode === 'full-audio' && (
            <div>
              <h2 className="card-title">
                <Headphones size={20} color="var(--accent)" /> Audiobook / Full Document Audio
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1.25rem' }}>
                Converts the <strong>exact text</strong> of your document directly into natural spoken voice. You can listen in the original language or have the full document translated first.
              </p>

              <div className="controls-grid">
                <div className="form-group">
                  <label>Read In Language</label>
                  <select
                    value={fullAudioLang}
                    onChange={(e) => setFullAudioLang(e.target.value)}
                  >
                    <option value="Original">Original ({uploadData.detected_language})</option>
                    {health?.supported_languages?.map((lang) => (
                      <option key={lang} value={lang}>
                        {lang}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <button
                className="btn btn-primary"
                onClick={handleGenerateFullDocumentAudio}
                disabled={isGeneratingFullAudio}
              >
                {isGeneratingFullAudio ? (
                  <>
                    <div className="spinner" /> Generating Full Document Audio ({fullAudioLang})...
                  </>
                ) : (
                  <>
                    <Volume2 size={18} /> Generate Full Document Audio ({fullAudioLang})
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      )}

      {/* 3A: AI Summary Result Card */}
      {activeMode === 'summary' && summaryData && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h2 className="card-title" style={{ margin: 0 }}>
              <FileText size={20} color="var(--accent)" /> Final Summary ({summaryData.target_language})
            </h2>
            <button className="btn btn-secondary" onClick={handleCopySummary} style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
              <Copy size={14} /> {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>

          {summaryData.mock && (
            <div className="alert alert-warning">
              <AlertCircle size={16} />
              <span>Generated in development mock mode (quota preservation).</span>
            </div>
          )}

          <div className="summary-box">
            {summaryData.summary?.replace(/^\[MOCK SUMMARY.*?\]\s*/i, '')}
          </div>

          {/* Audio section */}
          <div style={{ marginTop: '1.5rem' }}>
            {!audioUrl ? (
              <button
                className="btn btn-secondary"
                onClick={handleGenerateAudio}
                disabled={isGeneratingAudio}
                style={{ width: '100%' }}
              >
                {isGeneratingAudio ? (
                  <>
                    <div className="spinner" /> Generating Voice Output ({targetLanguage})...
                  </>
                ) : (
                  <>
                    <Volume2 size={18} /> Convert Summary to Speech ({targetLanguage})
                  </>
                )}
              </button>
            ) : (
              <div className="audio-card">
                <audio controls autoPlay src={audioUrl}>
                  Your browser does not support the audio element.
                </audio>
                <a
                  href={`${audioUrl}?download=true`}
                  download={audioFilename}
                  className="btn btn-secondary"
                  style={{ textDecoration: 'none' }}
                >
                  <Download size={16} /> Download MP3
                </a>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 3B: Full Document Audio Result Card */}
      {activeMode === 'full-audio' && fullAudioData && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h2 className="card-title" style={{ margin: 0 }}>
              <Headphones size={20} color="var(--accent)" /> Full Document Audio ({fullAudioData.language})
            </h2>
            <button className="btn btn-secondary" onClick={handleCopyFullText} style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
              <Copy size={14} /> {copiedFullText ? 'Copied Full Text!' : 'Copy Full Text'}
            </button>
          </div>

          {/* Full Audio Player */}
          <div className="audio-card" style={{ marginTop: 0, marginBottom: '1.5rem' }}>
            <audio controls autoPlay src={fullAudioData.audio_url}>
              Your browser does not support the audio element.
            </audio>
            <a
              href={`${fullAudioData.audio_url}?download=true`}
              download={fullAudioData.filename}
              className="btn btn-secondary"
              style={{ textDecoration: 'none' }}
            >
              <Download size={16} /> Download Full MP3
            </a>
          </div>

          <h3 style={{ fontSize: '1rem', marginBottom: '0.75rem', color: 'var(--text-muted)' }}>
            Document Text ({fullAudioData.character_count.toLocaleString()} characters)
          </h3>
          <div className="summary-box" style={{ maxHeight: '350px', overflowY: 'auto' }}>
            {fullAudioData.full_text}
          </div>
        </div>
      )}
    </div>
  )
}
