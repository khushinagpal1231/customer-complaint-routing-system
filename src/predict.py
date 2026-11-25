"""Single entry point used by the app and the CLI.
CLI:  python -m src.predict "I was charged a fee I never agreed to..."
"""
import sys
import joblib
import pandas as pd
from . import config as C
from .preprocess import full_clean, light_clean
from .sentiment import sentiment_one, sentiment_band, using_transformer
from .priority import compute_priority

_model = None


def get_model():
    global _model
    if _model is None:
        p = C.MODEL_DIR / "deploy_tfidf_lr.pkl"
        if not p.exists():
            raise FileNotFoundError("Model not trained yet. Run: python -m src.train_baselines")
        _model = joblib.load(p)
    return _model


def predict_department(text: str, top_k: int = 3):
    m = get_model()
    proba = m.predict_proba([full_clean(text)])[0]
    order = proba.argsort()[::-1][:top_k]
    return [(m.classes_[i], float(proba[i])) for i in order]


def analyze(text: str, with_sentiment: bool = True, use_transformer: bool = True):
    ranked = predict_department(text)
    label, neg = sentiment_one(light_clean(text), use_transformer=use_transformer) if with_sentiment else ("n/a", 0.5)
    prio, score, hits = compute_priority(text, neg)
    return {
        "department": ranked[0][0], "confidence": ranked[0][1], "top3": ranked,
        "sentiment": label, "negativity": neg, "severity": sentiment_band(neg),
        "priority": prio, "priority_score": score, "keywords": hits,
    }


if __name__ == "__main__":
    txt = " ".join(sys.argv[1:]) or "A debt collector keeps calling me and threatening legal action for a debt that is not mine."
    r = analyze(txt)
    print(f"Department : {r['department']} ({r['confidence']:.0%})")
    print(f"Sentiment  : {r['sentiment']} (negativity {r['negativity']:.2f}, {r['severity']})")
    print(f"Priority   : {r['priority']} (score {r['priority_score']}, keywords {r['keywords']})")
