"""
segmentation.py  (UPGRADED)
- Semantic boundary detection using sentence similarity (optional, falls back gracefully)
- Better topic labelling using top keywords
- Configurable thresholds
"""

from src.keyword_extraction import extract_keywords


def _topic_label_from_keywords(keywords: list[str]) -> str:
    """Turn top keywords into a readable topic label."""
    if not keywords:
        return "General Discussion"
    # Capitalise and join top 3
    label = " · ".join(k.title() for k in keywords[:3])
    return label


def segment_transcript(
    transcript: dict,
    max_segment_duration: float = 120.0,   # seconds
    max_sentences: int = 8,
    min_segment_duration: float = 10.0,    # avoid tiny leftover segments
    use_semantic: bool = False             # set True if sentence-transformers installed
) -> list[dict]:
    """
    Segments a Whisper transcript into meaningful chunks.

    Strategy:
      1. Group Whisper segments by time (max_segment_duration) and
         sentence count (max_sentences) — same as before but smarter.
      2. If use_semantic=True, also try to detect topic shifts using
         cosine similarity between sentence embeddings (needs sentence-transformers).
      3. Each output segment gets: id, topic label, start/end times,
         full text, summary, keywords.

    Returns list of segment dicts.
    """
    from src.transcript import summarize_text

    whisper_segments = transcript["segments"]

    if not whisper_segments:
        return []

    # --- Optional semantic boundary detection ---
    semantic_boundaries = set()
    if use_semantic:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            st_model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [s["text"] for s in whisper_segments]
            embeddings = st_model.encode(texts, convert_to_numpy=True)

            for i in range(1, len(embeddings)):
                # cosine similarity between adjacent segments
                a, b = embeddings[i - 1], embeddings[i]
                sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                if sim < 0.35:   # low similarity → topic shift
                    semantic_boundaries.add(i)
        except ImportError:
            pass   # silently skip if not installed

    # --- Build segments ---
    segments = []
    current_text = []
    start_time = None
    sentence_count = 0

    for idx, seg in enumerate(whisper_segments):
        if start_time is None:
            start_time = seg["start"]

        current_text.append(seg["text"])
        sentence_count += 1

        duration = seg["end"] - start_time
        is_last = idx == len(whisper_segments) - 1

        at_time_limit = duration >= max_segment_duration
        at_sentence_limit = sentence_count >= max_sentences
        at_semantic_boundary = (idx + 1) in semantic_boundaries
        at_end = is_last

        should_close = (
            at_time_limit
            or at_sentence_limit
            or at_semantic_boundary
            or at_end
        )

        # Don't close too early unless it's the actual end
        too_short = duration < min_segment_duration and not at_end

        if should_close and not too_short:
            combined_text = " ".join(current_text)
            keywords = extract_keywords(combined_text, top_n=6)

            segments.append({
                "id":       len(segments) + 1,
                "topic":    _topic_label_from_keywords(keywords),
                "start":    round(start_time, 2),
                "end":      round(seg["end"], 2),
                "duration": round(seg["end"] - start_time, 2),
                "text":     combined_text,
                "summary":  summarize_text(combined_text, max_sentences=2),
                "keywords": keywords,
            })

            current_text = []
            start_time = None
            sentence_count = 0

    return segments
