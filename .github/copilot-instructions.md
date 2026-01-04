# Copilot Instructions for Document Extractor

## Project Overview
This is a Flask web application for document processing: upload files (PDF, DOCX, TXT, images), extract text via OCR (Tesseract), identify keywords/entities using spaCy and regex, classify documents (e.g., Education, Financial), generate extractive/abstractive summaries, translate content, and produce audio narration via gTTS.

## Architecture
- **Single Flask app** (`app.py`): Handles uploads, processing, and rendering.
- **Data Flow**: Upload → Text Extraction → Keyword Extraction → Classification → Summarization → Translation → Audio Generation → Template Render.
- **File Organization**: Uploads categorized into `uploads/{category}/` (e.g., Education, General). Audio files in `static/audio/`.
- **Key Functions**:
  - `extract_text()`: Handles PDF (fitz), DOCX (python-docx), TXT, images (OCR with cv2 + pytesseract).
  - `extract_keywords()`: Uses spaCy NER + regex for entities like PAN, AADHAAR, phones.
  - `detect_document_field()`: Classifies based on keyword patterns (e.g., education terms → "Education").
  - `extractive_summary()` / `abstractive_summary()`: Sumy LSA / Transformers pipeline.
  - `speak_text_dynamic()`: Generates unique MP3 files in `static/audio/` using gTTS.

## Setup & Dependencies
- **Python Libraries**: flask, fitz, docx, spacy (en_core_web_sm), pytesseract, cv2, PIL, numpy, deep_translator, nltk, sumy, transformers, gtts.
- **External Tools**: Tesseract OCR (path hardcoded to `C:\Program Files\Tesseract-OCR\tesseract.exe`).
- **Run**: `python app.py` (debug=True). Ensure Tesseract installed and path correct.
- **Models**: Download spaCy model `python -m spacy download en_core_web_sm`. Transformers model loads on first use.

## Conventions & Patterns
- **Entity Schema**: `ENTITY_SCHEMA` dict defines categories (IDENTITY, FINANCE, etc.) with keywords for classification.
- **Text Cleaning**: `clean_list()` removes duplicates/whitespace. Filters entities (e.g., PERSON excludes digit-containing or >4 words).
- **Regex Patterns**: Specific for Indian IDs (PAN: `[A-Z]{5}[0-9]{4}[A-Z]`, AADHAAR: `\d{4}\s\d{4}\s\d{4}`), phones, money.
- **Audio Naming**: Unique filenames with UUID (e.g., `audio_{uuid}.mp3`) to avoid conflicts.
- **File Serving**: `/uploads/<path:filename>` route serves categorized files; TXT shown inline, others downloaded.
- **Error Handling**: Wrap processing in try-except; return error strings for summaries/translations.
- **Translation**: Use `GoogleTranslator` for non-English targets; translate full text, keywords, summaries separately.

## Key Files
- `app.py`: Main app logic.
- `templates/index.html`: Renders results, audio players.
- `static/audio/`: TTS outputs.
- `uploads/`: Categorized documents.

## Workflows
- **Upload Processing**: Save temp, extract text, classify, move to category folder, generate outputs.
- **Debugging**: Check Tesseract path, model loads, audio file generation.
- **Extensions**: Add new entity types to `ENTITY_SCHEMA`; extend `extract_text()` for formats; integrate new summarizers.