# Multimodal Video & PDF Knowledge Distiller ⚡

An AI/ML-powered local-first application built with Streamlit, RAG (Retrieval-Augmented Generation), and Google Gemini API (`google-genai`). It converts dense PDF documents and YouTube video links into high-yield executive summaries, interactive study flashcards, and grounded Q&A support.

---

## ✨ Features

- 📄 **PDF Processing:** Extracts text from structured PDF files using `pdfplumber` with heuristic accuracy scoring.
- 📺 **YouTube Subtitle & Audio Pipeline:** Uses `youtube_transcript_api` for instant transcript retrieval, with a fallback to `yt-dlp` and local `OpenAI Whisper` speech-to-text models.
- ⚡ **Google Gemini Interactions API:** Uses the `google-genai` SDK and `gemini-3.8-flash` model for instant summarizing, flashcard extraction, and Q&A.
- 🔍 **Local Vector Search:** Employs `ChromaDB` for local vector storage and contextual retrieval.

---

## 🛠️ Project Structure

```text
knowledge_distiller/
├── modules/
│   ├── extractor.py       # PDF parsing & YouTube subtitle/Whisper pipeline
│   ├── rag_engine.py      # Google Gemini API integration
│   └── vector_store.py    # ChromaDB indexing & semantic retrieval
├── app.py                 # Streamlit UI application
├── config.py              # Configuration & API key handling
├── requirements.txt       # Dependencies
├── .env.example           # API Key template
├── .gitignore             # Files excluded from version control
└── README.md              # Project documentation