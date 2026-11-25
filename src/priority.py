"""Rule-based priority. NOTE: the dataset has NO priority label, so this is a
documented HEURISTIC, not a learned model. Say this clearly in the report/viva."""
import re

URGENT_KEYWORDS = [
    "fraud", "stolen", "urgent", "immediately", "legal", "lawyer", "attorney",
    "lawsuit", "sue", "scam", "unauthorized", "identity theft", "harass",
    "threat", "foreclosure", "eviction", "garnish", "police", "court",
]


def keyword_hits(text: str):
    t = text.lower()
    return [k for k in URGENT_KEYWORDS if re.search(r"\b" + re.escape(k), t)]


def compute_priority(text: str, neg_confidence: float = 0.5):
    """Return (label, score 0-100, matched_keywords).
    score = 40*negativity + 15 per keyword (max 45) + 15 if long complaint."""
    hits = keyword_hits(text)
    score = 40 * float(neg_confidence)
    score += min(len(hits) * 15, 45)
    if len(text.split()) > 150:
        score += 15
    score = min(score, 100)
    label = "High" if score >= 60 else "Medium" if score >= 35 else "Low"
    return label, round(score, 1), hits
