"""Step 5: run DistilBERT sentiment + priority on a sample of the test set and chart it.
Run: python -m src.sentiment_analysis   (needs transformers + torch; else uses fallback)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C
from .utils import load_split
from .sentiment import sentiment_many, using_transformer
from .priority import compute_priority


def main(n=1000):
    _, test = load_split()
    s = test.sample(min(n, len(test)), random_state=C.SEED).copy()
    res = sentiment_many(s["text_light"].tolist())
    s["sentiment"] = [r[0] for r in res]
    s["neg_conf"] = [r[1] for r in res]
    pr = [compute_priority(t, nc) for t, nc in zip(s["text_light"], s["neg_conf"])]
    s["priority"] = [p[0] for p in pr]; s["priority_score"] = [p[1] for p in pr]
    s.to_csv(C.RES_DIR / "sentiment_priority_sample.csv", index=False)
    print("Transformer used:", using_transformer())

    s["sentiment"].value_counts().plot.bar(rot=0, color="#ef4444"); plt.title("Sentiment of complaints")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "sentiment_distribution.png", dpi=130); plt.close()
    pd.crosstab(s["product"], s["priority"]).plot.bar(stacked=True, figsize=(9, 5), rot=25)
    plt.title("Priority by department (rule-based)"); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "priority_by_department.png", dpi=130); plt.close()
    print(s["priority"].value_counts())


if __name__ == "__main__":
    main()
