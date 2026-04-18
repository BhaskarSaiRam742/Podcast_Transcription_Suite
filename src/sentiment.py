"""
sentiment.py  (NEW)
Per-segment sentiment analysis using VADER (fast, no GPU needed).
Optionally upgrades to a transformer model if available.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


def _vader_sentiment(text: str) -> dict:
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
        emoji = "😊"
    elif compound <= -0.05:
        label = "Negative"
        emoji = "😟"
    else:
        label = "Neutral"
        emoji = "😐"

    return {
        "label":    label,
        "emoji":    emoji,
        "score":    round(compound, 3),
        "positive": round(scores["pos"], 3),
        "neutral":  round(scores["neu"], 3),
        "negative": round(scores["neg"], 3),
    }


def analyze_sentiment(text: str, use_transformer: bool = False) -> dict:
    """
    Analyze sentiment of a text block.

    Args:
        text:            Input text
        use_transformer: If True and transformers is installed, use
                         distilbert-base-uncased-finetuned-sst-2-english
                         for more accurate results (slower first run).

    Returns dict with keys: label, emoji, score, positive, neutral, negative
    """
    if not text or len(text.strip()) < 5:
        return {"label": "Neutral", "emoji": "😐", "score": 0.0,
                "positive": 0.0, "neutral": 1.0, "negative": 0.0}

    if use_transformer:
        try:
            from transformers import pipeline as hf_pipeline
            # Lazy-load and cache
            if not hasattr(analyze_sentiment, "_transformer"):
                analyze_sentiment._transformer = hf_pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True,
                    max_length=512
                )
            result = analyze_sentiment._transformer(text[:512])[0]
            label = result["label"].capitalize()
            score = round(result["score"], 3)
            emoji = "😊" if label == "Positive" else "😟"
            return {"label": label, "emoji": emoji, "score": score,
                    "positive": score if label == "Positive" else round(1 - score, 3),
                    "neutral": 0.0,
                    "negative": score if label == "Negative" else round(1 - score, 3)}
        except ImportError:
            pass   # fall back to VADER

    return _vader_sentiment(text)


def add_sentiment_to_segments(segments: list[dict], use_transformer: bool = False) -> list[dict]:
    """Add a 'sentiment' key to each segment dict in-place. Returns segments."""
    for seg in segments:
        seg["sentiment"] = analyze_sentiment(seg.get("text", ""), use_transformer)
    return segments
