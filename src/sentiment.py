"""Pretrained DistilBERT sentiment (no training needed). Falls back to a tiny
lexicon if `transformers` is not installed, so the app never crashes."""
from functools import lru_cache
from .config import SENTIMENT_MODEL

NEG_WORDS = {"fraud", "scam", "terrible", "worst", "angry", "unauthorized", "stolen",
             "harass", "unacceptable", "rude", "never", "refuse", "lied", "illegal"}


@lru_cache(maxsize=1)
def _pipe():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis", model=SENTIMENT_MODEL, truncation=True, max_length=512)
    except Exception:
        return None


def using_transformer() -> bool:
    return _pipe() is not None


def _lexicon_sentiment(text: str):
    words = set(text.lower().split())
    neg = min(0.5 + 0.1 * len(words & NEG_WORDS), 0.99)
    return "Negative (lexicon fallback)", neg


def sentiment_one(text: str, use_transformer: bool = True):
    """Return (label, negative_confidence 0-1)."""
    if use_transformer:
        p = _pipe()
        if p is not None:
            r = p(text[:2000])[0]
            neg = r["score"] if r["label"] == "NEGATIVE" else 1 - r["score"]
            return r["label"].capitalize(), float(neg)
    return _lexicon_sentiment(text)


def sentiment_many(texts, batch_size=32):
    p = _pipe()
    if p is None:
        return [sentiment_one(t) for t in texts]
    out = p([t[:2000] for t in texts], batch_size=batch_size)
    return [(r["label"].capitalize(), float(r["score"] if r["label"] == "NEGATIVE" else 1 - r["score"])) for r in out]


def sentiment_band(neg: float) -> str:
    """Complaints are almost all negative, so use the confidence as a severity level."""
    return "Very negative" if neg >= 0.95 else "Negative" if neg >= 0.7 else "Mixed/Neutral" if neg >= 0.4 else "Positive"
