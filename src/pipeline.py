"""
pipeline.py  (NEW)
Central orchestrator — called by app.py (Streamlit) and usable as a library.
Handles both uploaded files and URLs uniformly.
"""

import os
import tempfile
from typing import Callable


def run_pipeline(
    source: str | os.PathLike,
    source_type: str = "file",          # "file" | "url"
    use_semantic_segmentation: bool = False,
    use_transformer_sentiment: bool = False,
    whisper_language: str = None,
    output_dir: str = "audio_processed",
    progress_cb: Callable[[str, int], None] = None,
) -> dict:
    """
    Full pipeline: preprocess → transcribe → segment → sentiment.

    Args:
        source:                    local file path or URL string
        source_type:               "file" or "url"
        use_semantic_segmentation: enable sentence-transformer topic detection
        use_transformer_sentiment: use distilbert instead of VADER
        whisper_language:          ISO code or None (auto-detect)
        output_dir:                directory for temp audio files
        progress_cb:               optional callback(message, percent) for UI

    Returns big result dict consumed by the Streamlit UI.
    """
    def _progress(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    transcript = None
    source_label = "Uploaded file"
    skipped_preprocessing = False

    # ── Step 1: Resolve source ────────────────────────────────────────────────
    if source_type == "url":
        _progress("🔗 Resolving URL…", 5)
        from src.url_handler import resolve_url
        resolved = resolve_url(str(source), output_dir)
        source_label = resolved["source_label"]

        if resolved["type"] == "transcript":
            # Existing transcript found — skip audio pipeline
            transcript = resolved["data"]
            skipped_preprocessing = True
            _progress(f"✅ Existing transcript found ({source_label})", 40)
        else:
            source = resolved["file_path"]

    # ── Step 2: Preprocess audio (skip if transcript already found) ───────────
    if not skipped_preprocessing:
        _progress("🔊 Preprocessing audio…", 15)
        from src.preprocess_audio import preprocess_audio
        cleaned_path = preprocess_audio(str(source), output_dir)
        _progress("✅ Audio preprocessed", 30)

        # ── Step 3: Transcribe ─────────────────────────────────────────────────
        _progress("📝 Transcribing (this may take a while)…", 35)
        from src.transcript import transcribe_audio
        transcript = transcribe_audio(cleaned_path, language=whisper_language)
        _progress("✅ Transcription complete", 60)
    else:
        # Still need a summary for YouTube caption transcripts
        from src.transcript import summarize_text
        if not transcript.get("summary"):
            transcript["summary"] = summarize_text(transcript["full_text"])

    # ── Step 4: Segmentation ──────────────────────────────────────────────────
    _progress("🔀 Segmenting transcript…", 65)
    from src.segmentation import segment_transcript
    segments = segment_transcript(
        transcript,
        use_semantic=use_semantic_segmentation
    )
    _progress("✅ Segmentation complete", 80)

    # ── Step 5: Sentiment Analysis ────────────────────────────────────────────
    _progress("💬 Analysing sentiment…", 85)
    from src.sentiment import add_sentiment_to_segments
    segments = add_sentiment_to_segments(segments, use_transformer=use_transformer_sentiment)
    _progress("✅ Sentiment analysis complete", 95)

    _progress("🎉 Done!", 100)

    return {
        "source_label":           source_label,
        "skipped_preprocessing":  skipped_preprocessing,
        "full_text":              transcript["full_text"],
        "summary":                transcript["summary"],
        "language":               transcript.get("language", "unknown"),
        "transcript_source":      transcript.get("source", "whisper"),
        "segments":               segments,
        "segment_count":          len(segments),
    }
