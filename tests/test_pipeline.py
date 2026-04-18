"""
tests/test_pipeline.py
Unit tests for all src modules.
Run with:  pytest tests/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
# keyword_extraction
# ─────────────────────────────────────────────────────────────────────────────
class TestKeywordExtraction:
    def test_basic_extraction(self):
        from src.keyword_extraction import extract_keywords
        text = "Machine learning and artificial intelligence are transforming the technology industry rapidly."
        kws = extract_keywords(text, top_n=4)
        assert isinstance(kws, list)
        assert len(kws) <= 4
        assert all(isinstance(k, str) for k in kws)

    def test_empty_text_returns_empty(self):
        from src.keyword_extraction import extract_keywords
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []
        assert extract_keywords("hi") == []

    def test_no_duplicate_substrings(self):
        from src.keyword_extraction import extract_keywords
        text = "climate change and climate policy and climate science are all related to climate."
        kws = extract_keywords(text, top_n=6)
        # No keyword should be a substring of another
        for i, a in enumerate(kws):
            for j, b in enumerate(kws):
                if i != j:
                    assert a not in b and b not in a, f"Duplicate substring: '{a}' vs '{b}'"

    def test_with_scores(self):
        from src.keyword_extraction import extract_keywords_with_scores
        text = "deep learning neural networks have revolutionized computer vision tasks."
        results = extract_keywords_with_scores(text, top_n=3)
        assert all(isinstance(word, str) and isinstance(score, float) for word, score in results)
        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# sentiment
# ─────────────────────────────────────────────────────────────────────────────
class TestSentiment:
    def test_positive_text(self):
        from src.sentiment import analyze_sentiment
        result = analyze_sentiment("This is absolutely amazing and wonderful!")
        assert result["label"] == "Positive"
        assert result["score"] > 0

    def test_negative_text(self):
        from src.sentiment import analyze_sentiment
        result = analyze_sentiment("This is terrible, awful and completely broken.")
        assert result["label"] == "Negative"
        assert result["score"] < 0

    def test_neutral_text(self):
        from src.sentiment import analyze_sentiment
        result = analyze_sentiment("The meeting is on Tuesday at 3pm.")
        assert result["label"] == "Neutral"

    def test_empty_text(self):
        from src.sentiment import analyze_sentiment
        result = analyze_sentiment("")
        assert result["label"] == "Neutral"
        assert result["score"] == 0.0

    def test_required_keys(self):
        from src.sentiment import analyze_sentiment
        result = analyze_sentiment("Some text here.")
        for key in ("label", "emoji", "score", "positive", "neutral", "negative"):
            assert key in result

    def test_add_sentiment_to_segments(self):
        from src.sentiment import add_sentiment_to_segments
        segments = [
            {"id": 1, "text": "Great show today!"},
            {"id": 2, "text": "The weather is okay."},
        ]
        result = add_sentiment_to_segments(segments)
        assert all("sentiment" in s for s in result)
        assert result[0]["sentiment"]["label"] == "Positive"


# ─────────────────────────────────────────────────────────────────────────────
# transcript (summarize_text only — no audio file needed)
# ─────────────────────────────────────────────────────────────────────────────
class TestTranscript:
    def test_summarize_short_text(self):
        from src.transcript import summarize_text
        text = "This is sentence one. This is sentence two."
        result = summarize_text(text)
        assert len(result) > 0
        assert result[-1] in ".!?"

    def test_summarize_long_text(self):
        from src.transcript import summarize_text
        sentences = [f"This is sentence number {i}." for i in range(20)]
        text = " ".join(sentences)
        result = summarize_text(text, max_sentences=4)
        # Should not return all 20 sentences
        result_sentences = result.split(". ")
        assert len(result_sentences) <= 6   # small buffer

    def test_summarize_empty(self):
        from src.transcript import summarize_text
        assert summarize_text("") == ""


# ─────────────────────────────────────────────────────────────────────────────
# segmentation
# ─────────────────────────────────────────────────────────────────────────────
class TestSegmentation:
    def _make_transcript(self, n_segments=20, seg_duration=10):
        """Build a fake Whisper transcript dict."""
        segments = []
        for i in range(n_segments):
            segments.append({
                "start": i * seg_duration,
                "end": (i + 1) * seg_duration,
                "text": f"This is segment {i} discussing important topics about technology and innovation."
            })
        full_text = " ".join(s["text"] for s in segments)
        return {"full_text": full_text, "summary": "", "segments": segments}

    def test_segments_are_ordered(self):
        from src.segmentation import segment_transcript
        transcript = self._make_transcript(30)
        segs = segment_transcript(transcript)
        starts = [s["start"] for s in segs]
        assert starts == sorted(starts)

    def test_no_overlapping_segments(self):
        from src.segmentation import segment_transcript
        transcript = self._make_transcript(30)
        segs = segment_transcript(transcript)
        for i in range(len(segs) - 1):
            assert segs[i]["end"] <= segs[i + 1]["start"] + 0.01  # tiny float tolerance

    def test_all_text_covered(self):
        from src.segmentation import segment_transcript
        transcript = self._make_transcript(20)
        segs = segment_transcript(transcript)
        combined = " ".join(s["text"] for s in segs)
        # Every word from original should be in the combined output
        original_words = set(transcript["full_text"].lower().split())
        combined_words = set(combined.lower().split())
        assert original_words == combined_words

    def test_segment_has_required_keys(self):
        from src.segmentation import segment_transcript
        transcript = self._make_transcript(10)
        segs = segment_transcript(transcript)
        for seg in segs:
            for key in ("id", "topic", "start", "end", "duration", "text", "keywords", "summary"):
                assert key in seg, f"Missing key: {key}"

    def test_empty_transcript(self):
        from src.segmentation import segment_transcript
        result = segment_transcript({"segments": [], "full_text": "", "summary": ""})
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# url_handler (unit-level, no network calls)
# ─────────────────────────────────────────────────────────────────────────────
class TestUrlHandler:
    def test_youtube_detection(self):
        from src.url_handler import is_youtube_url
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert is_youtube_url("https://www.youtube.com/shorts/abc123")
        assert not is_youtube_url("https://vimeo.com/123456")
        assert not is_youtube_url("https://example.com/podcast.mp3")

    def test_direct_media_detection(self):
        from src.url_handler import is_direct_media_url
        assert is_direct_media_url("https://example.com/episode.mp3")
        assert is_direct_media_url("https://cdn.example.com/show.mp4")
        assert is_direct_media_url("https://files.example.com/audio.wav?token=abc")
        assert not is_direct_media_url("https://www.youtube.com/watch?v=abc")
        assert not is_direct_media_url("https://example.com/rss")


# ─────────────────────────────────────────────────────────────────────────────
# export
# ─────────────────────────────────────────────────────────────────────────────
class TestExport:
    def _make_result(self):
        return {
            "source_label": "Test source",
            "language": "en",
            "transcript_source": "whisper",
            "full_text": "Hello world. This is a test transcript.",
            "summary": "A test transcript.",
            "segment_count": 2,
            "segments": [
                {
                    "id": 1, "topic": "Intro · Hello World",
                    "start": 0.0, "end": 30.0, "duration": 30.0,
                    "text": "Hello world.",
                    "summary": "Hello world.",
                    "keywords": ["hello", "world"],
                    "sentiment": {"label": "Positive", "emoji": "😊", "score": 0.5,
                                  "positive": 0.5, "neutral": 0.4, "negative": 0.1}
                },
                {
                    "id": 2, "topic": "Test · Transcript",
                    "start": 30.0, "end": 60.0, "duration": 30.0,
                    "text": "This is a test transcript.",
                    "summary": "A test.",
                    "keywords": ["test", "transcript"],
                    "sentiment": {"label": "Neutral", "emoji": "😐", "score": 0.0,
                                  "positive": 0.1, "neutral": 0.8, "negative": 0.1}
                },
            ]
        }

    def test_json_export(self):
        from src.export import to_json
        import json
        result = self._make_result()
        output = to_json(result)
        parsed = json.loads(output)
        assert parsed["segment_count"] == 2

    def test_srt_export(self):
        from src.export import to_srt
        result = self._make_result()
        srt = to_srt(result["segments"])
        assert "00:00:00,000 --> 00:00:30,000" in srt
        assert "Hello world." in srt

    def test_txt_export(self):
        from src.export import to_txt
        result = self._make_result()
        txt = to_txt(result)
        assert "SUMMARY" in txt
        assert "Hello world." in txt

    def test_csv_export(self):
        from src.export import to_csv
        import csv, io
        result = self._make_result()
        csv_str = to_csv(result["segments"])
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["sentiment_label"] == "Positive"

    def test_markdown_export(self):
        from src.export import to_markdown
        result = self._make_result()
        md = to_markdown(result)
        assert "# 🎙️ Podcast Transcript" in md
        assert "## Summary" in md
        assert "## Segments" in md
