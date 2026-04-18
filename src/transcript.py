"""
transcript.py  (SPEED + ACCURACY OPTIMISED)
- Uses faster-whisper instead of openai-whisper (4x faster on CPU)
- tiny model + int8 quantisation = very fast, low RAM
- beam_size=5 + best_of=5 = recovers accuracy lost from smaller model
- vad_filter=True = skips silence automatically before processing
- condition_on_previous_text=True = maintains context between chunks
- summarize_text() exported for YouTube caption transcripts
"""

import re

# ── Model settings ────────────────────────────────────────────────────────────
# "tiny"  : ~75 MB,  fastest,  use this for CPU (your current setup)
# "base"  : ~145 MB, 2x slower than tiny, more accurate
# "small" : ~465 MB, 4x slower than tiny, good balance
_MODEL_SIZE = "tiny"

_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            _MODEL_SIZE,
            device="cpu",
            compute_type="int8",      # quantises weights: 4x less RAM, faster
            num_workers=4,            # parallel data loading
            cpu_threads=0,            # 0 = use all available CPU cores
        )
    return _model


def summarize_text(text: str, max_sentences: int = 4) -> str:
    """
    Extractive summary: first 2 + last 2 sentences.
    Gives a start-and-conclusion feel without any AI model.
    """
    if not text:
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) <= max_sentences:
        summary = " ".join(sentences)
    else:
        half = max_sentences // 2
        picked = sentences[:half] + sentences[-(max_sentences - half):]
        summary = " ".join(picked)

    if summary and summary[-1] not in ".!?":
        summary += "."

    return summary


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Transcribe a local audio file using faster-whisper.

    Key accuracy-preserving settings even with tiny model:
      - beam_size=5       : explores 5 hypothesis paths (same as medium default)
      - best_of=5         : picks best result from 5 temperature samples
      - vad_filter=True   : strips silence before processing (speeds up + accuracy)
      - condition_on_previous_text=True : uses previous chunk as context

    Args:
        audio_path : path to cleaned .wav file
        language   : ISO 639-1 code e.g. "en", or None for auto-detect

    Returns:
        {
          "full_text": str,
          "summary":   str,
          "segments":  [{"start", "end", "text"}, ...],
          "language":  str,
          "source":    "faster-whisper-tiny"
        }
    """
    model = get_model()

    options = dict(
        beam_size=5,                      # accuracy: explore more paths
        best_of=5,                        # accuracy: sample 5 times, pick best
        patience=1.5,                     # accuracy: wait longer before stopping beam
        vad_filter=True,                  # speed: skip silence automatically
        vad_parameters=dict(
            min_silence_duration_ms=500,  # ignore silences shorter than 0.5s
            speech_pad_ms=400,            # keep 400ms buffer around speech
        ),
        condition_on_previous_text=True,  # accuracy: use context between chunks
        no_speech_threshold=0.6,          # skip chunks that are probably not speech
        log_prob_threshold=-1.0,          # retry chunks with low confidence
        compression_ratio_threshold=2.4,  # detect and skip repetitive/hallucinated text
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # fallback temps if confidence low
        word_timestamps=False,            # skip word-level (saves time)
    )

    if language:
        options["language"] = language

    raw_segments, info = model.transcribe(audio_path, **options)

    # faster-whisper returns a generator — consume it fully
    segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        if text:                          # skip empty segments
            segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end,   2),
                "text":  text,
            })

    full_text = " ".join(s["text"] for s in segments)

    return {
        "full_text": full_text,
        "summary":   summarize_text(full_text),
        "segments":  segments,
        "language":  info.language,
        "source":    f"faster-whisper-{_MODEL_SIZE}",
    }
