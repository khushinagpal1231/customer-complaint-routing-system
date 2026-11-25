"""Step 6: find wrong predictions and the most confused department pairs.
Run: python -m src.error_analysis
Then OPEN outputs/results/errors_for_review.csv and fill the 'my_reason' column yourself.
"""
import joblib
import pandas as pd
from . import config as C
from .utils import load_split


def main(n=20):
    _, test = load_split()
    pipe = joblib.load(C.MODEL_DIR / "deploy_tfidf_lr.pkl")
    proba = pipe.predict_proba(test["text_clean"])
    test = test.copy()
    test["pred"] = pipe.classes_[proba.argmax(1)]
    test["confidence"] = proba.max(1).round(3)
    test["n_words"] = test["text_light"].str.split().str.len()
    wrong = test[test["pred"] != test["product"]]
    right = test[test["pred"] == test["product"]]

    pairs = wrong.groupby(["product", "pred"]).size().sort_values(ascending=False).reset_index(name="count")
    pairs.to_csv(C.RES_DIR / "confused_pairs.csv", index=False)
    print("Most confused pairs (true -> predicted):"); print(pairs.head(8).to_string(index=False))

    cols = ["product", "pred", "confidence", "n_words", "issue", "text_light"]
    out = wrong.sort_values("confidence", ascending=False).head(n)[cols].copy()
    out["my_reason"] = ""    # <- YOU fill this: overlap of categories? short text? multi-topic? masked data?
    out.to_csv(C.RES_DIR / "errors_for_review.csv", index=False)
    right.sample(min(10, len(right)), random_state=C.SEED)[cols].to_csv(C.RES_DIR / "correct_examples.csv", index=False)
    print(f"\nError rate: {len(wrong)/len(test):.1%}.  Avg words wrong={wrong['n_words'].mean():.0f}  right={right['n_words'].mean():.0f}")


if __name__ == "__main__":
    main()
