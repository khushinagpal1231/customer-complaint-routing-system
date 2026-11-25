"""Step 2: classical baselines (BoW / TF-IDF  x  NB / LR / SVM).
Run: python -m src.train_baselines
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from . import config as C
from .utils import load_split, metrics, plot_cm

EXPERIMENTS = {
    "BoW + NaiveBayes":     lambda: make_pipeline(CountVectorizer(max_features=30000, ngram_range=(1, 1)), MultinomialNB()),
    "TF-IDF + NaiveBayes":  lambda: make_pipeline(TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2), MultinomialNB()),
    "TF-IDF + LogReg":      lambda: make_pipeline(TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=1000, C=5)),
    "TF-IDF + LinearSVM":   lambda: make_pipeline(TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2), LinearSVC(C=0.5)),
}


def top_words(pipe, n=10):
    vec, clf = pipe.steps[0][1], pipe.steps[-1][1]
    names = np.array(vec.get_feature_names_out())
    rows = []
    for i, cls in enumerate(clf.classes_):
        idx = np.argsort(clf.coef_[i])[::-1][:n]
        rows.append({"department": cls, "top_words": ", ".join(names[idx])})
    return pd.DataFrame(rows)


def main():
    train, test = load_split()
    labels = sorted(train["product"].unique())
    rows = []
    for name, make in EXPERIMENTS.items():
        pipe = make().fit(train["text_clean"], train["product"])
        pred = pipe.predict(test["text_clean"])
        m = metrics(test["product"], pred); m["model"] = name; rows.append(m)
        print(f"{name:22s} acc={m['accuracy']:.3f}  macroF1={m['f1_macro']:.3f}")
        tag = name.replace(" + ", "_").replace(" ", "")
        joblib.dump(pipe, C.MODEL_DIR / f"{tag}.pkl")
        plot_cm(test["product"], pred, labels, f"Confusion matrix - {name}", f"cm_{tag}.png")
        if name == "TF-IDF + LogReg":
            joblib.dump(pipe, C.MODEL_DIR / "deploy_tfidf_lr.pkl")      # used by the app
            top_words(pipe).to_csv(C.RES_DIR / "top_words_per_department.csv", index=False)
            (C.RES_DIR / "classification_report_tfidf_lr.txt").write_text(classification_report(test["product"], pred))
    res = pd.DataFrame(rows)[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"]].round(4)
    res.to_csv(C.RES_DIR / "baseline_results.csv", index=False)

    res.set_index("model")[["accuracy", "f1_macro"]].plot.bar(figsize=(8, 4.5), rot=20)
    plt.ylim(0, 1); plt.title("Baseline comparison (test set)"); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "model_comparison_baselines.png", dpi=130); plt.close()
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
