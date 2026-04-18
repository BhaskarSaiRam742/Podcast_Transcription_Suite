# 🎙️ Podcast Transcription Suite

A full-stack podcast transcription pipeline with a Streamlit web UI.
Supports uploaded audio/video files **and** URLs (YouTube, direct links, RSS feeds).

---

## Features

| Feature | Status |
|---|---|
| Upload audio files (mp3, wav, m4a, ogg, flac, aac) | ✅ |
| Upload video files (mp4, mkv, avi, mov, webm) | ✅ NEW |
| YouTube URL — fetch existing captions | ✅ NEW |
| YouTube URL — download & transcribe if no captions | ✅ NEW |
| Direct audio/video URL | ✅ NEW |
| Podcast RSS feed (latest episode) | ✅ NEW |
| Audio preprocessing (noise reduction, silence removal) | ✅ Upgraded |
| Transcription via Whisper (medium model) | ✅ Upgraded |
| Time-based segmentation | ✅ Upgraded |
| Semantic segmentation (optional) | ✅ NEW |
| Per-segment keyword extraction | ✅ Improved |
| Per-segment sentiment analysis (VADER) | ✅ NEW |
| Advanced sentiment (distilbert, optional) | ✅ NEW |
| Streamlit Web UI | ✅ NEW |
| Export to JSON / SRT / TXT | ✅ NEW |

---

## Setup

### 1. Install system dependencies

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> For better sentiment analysis, also run:
> ```bash
> pip install transformers sentence-transformers
> ```

---

## Running the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Project Structure

```
podcast_transcriber/
├── app.py                    ← Streamlit entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── pipeline.py           ← Central orchestrator
│   ├── url_handler.py        ← YouTube / URL / RSS resolver
│   ├── preprocess_audio.py   ← Audio + video preprocessing
│   ├── transcript.py         ← Whisper transcription
│   ├── segmentation.py       ← Segment + topic labelling
│   ├── sentiment.py          ← Per-segment sentiment
│   └── keyword_extraction.py ← TF-IDF keywords
└── audio_processed/          ← Temp files (auto-created)
```

---

## Pipeline Flow

```
Input
 ├── Uploaded file (audio/video)
 └── URL
      ├── YouTube → check captions → skip to segmentation if found
      ├── Direct audio/video URL → download
      └── RSS feed → download latest episode
           ↓
    [Preprocess] noise reduction, silence removal, video→audio
           ↓
    [Transcribe] Whisper medium
           ↓
    [Segment] time + optional semantic boundaries
           ↓
    [Sentiment] VADER per segment
           ↓
    [UI] Streamlit dashboard
```

---

## Deploying online (Streamlit Community Cloud)

1. Push the project to a public GitHub repo.
2. Go to https://share.streamlit.io → "New app"
3. Select your repo, branch, and `app.py` as the entry file.
4. Add `ffmpeg` as a system package in `packages.txt`:
   ```
   ffmpeg
   ```
5. Deploy — you get a public URL.

> Note: Whisper "medium" model requires ~1.5 GB RAM. Use the free tier carefully;
> the "base" model is faster and fits within free-tier limits.
> Change `_MODEL_SIZE = "base"` in `src/transcript.py` if needed.
