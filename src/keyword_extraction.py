"""
keyword_extraction.py  (IMPROVED)
- Same TF-IDF approach but with better deduplication and stopword handling
- Returns keywords with their scores for use in UI
"""

from sklearn.feature_extraction.text import TfidfVectorizer


_EXTRA_STOPWORDS = {
    "yeah", "know", "like", "just", "going", "thing", "things", "said",
    "say", "think", "really", "actually", "kind", "lot", "good", "right",
    "okay", "oh", "well", "also", "want", "got", "get", "let", "make",
    "way", "time", "people", "little", "something", "talk", "talking"
}


def extract_keywords(text: str, top_n: int = 6) -> list[str]:
    """
    Extract top N keywords/keyphrases from text using TF-IDF.
    Returns a list of keyword strings.
    """
    if not text or len(text.strip()) < 10:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100,
        sublinear_tf=True,
    )

    try:
        tfidf = vectorizer.fit_transform([text])
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]

    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    keywords = []
    for word, score in ranked:
        word = word.lower().strip()

        # Skip extra stopwords
        if any(sw in word.split() for sw in _EXTRA_STOPWORDS):
            continue

        # Skip if it's a substring or superset of an existing keyword
        if any(word in k or k in word for k in keywords):
            continue

        keywords.append(word)

        if len(keywords) >= top_n:
            break

    return keywords


def extract_keywords_with_scores(text: str, top_n: int = 6) -> list[tuple[str, float]]:
    """Same as extract_keywords but returns (keyword, score) tuples."""
    if not text or len(text.strip()) < 10:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100,
        sublinear_tf=True,
    )

    try:
        tfidf = vectorizer.fit_transform([text])
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    result = []
    for word, score in ranked:
        word = word.lower().strip()
        if any(sw in word.split() for sw in _EXTRA_STOPWORDS):
            continue
        if any(word in k or k in word for k in [r[0] for r in result]):
            continue
        result.append((word, round(float(score), 4)))
        if len(result) >= top_n:
            break

    return result
