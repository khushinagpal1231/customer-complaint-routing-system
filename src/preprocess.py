"""Text cleaning. Two versions are produced:
 - text_light : only URLs / emails / XXXX masks removed (transformers need natural text)
 - text_clean : lowercase, no punctuation/numbers, no stop-words, lemmatized (BoW / TF-IDF / LDA)
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negations: they carry meaning in complaints ("did not", "never")
KEEP = {"not", "no", "never", "nor", "cannot"}
STOP = set(ENGLISH_STOP_WORDS) - KEEP

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
MASK_RE = re.compile(r"\b[xX]{2,}\b|\{\$[\d.,]+\}")   # CFPB masking: XXXX, {$12.50}
NON_ALPHA = re.compile(r"[^a-z\s]")

try:
    from nltk.stem import WordNetLemmatizer
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    _lem = WordNetLemmatizer()
    _lem.lemmatize("tests")
    def _lemma(w): return _lem.lemmatize(w)
except Exception:                       # no nltk / no wordnet -> light suffix stripping
    def _lemma(w):
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                return w[: -len(suf)]
        return w


def light_clean(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = MASK_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def full_clean(text: str) -> str:
    text = light_clean(text).lower()
    text = NON_ALPHA.sub(" ", text)
    tokens = [_lemma(t) for t in text.split() if t not in STOP and len(t) > 2]
    return " ".join(tokens)
