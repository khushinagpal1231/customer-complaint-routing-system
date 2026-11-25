"""Step 3: Sentence-BERT embeddings + Logistic Regression (transformer component).
Run: python -m src.train_sbert        (needs: pip install sentence-transformers torch)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from . import config as C
from .utils import load_split, metrics, plot_cm


def main():
    from sentence_transformers import SentenceTransformer
    train, test = load_split()
    labels = sorted(train["product"].unique())
    model = SentenceTransformer(C.SBERT_MODEL)
    model.max_seq_length = 256
    print("Encoding train ..."); Xtr = model.encode(train["text_light"].tolist(), batch_size=64, show_progress_bar=True)
    print("Encoding test ...");  Xte = model.encode(test["text_light"].tolist(), batch_size=64, show_progress_bar=True)
    np.save(C.MODEL_DIR / "sbert_train.npy", Xtr); np.save(C.MODEL_DIR / "sbert_test.npy", Xte)

    clf = LogisticRegression(max_iter=2000, C=5).fit(Xtr, train["product"])
    pred = clf.predict(Xte)
    m = metrics(test["product"], pred); m["model"] = "SBERT + LogReg"
    print(m)
    joblib.dump(clf, C.MODEL_DIR / "sbert_lr.pkl")
    plot_cm(test["product"], pred, labels, "Confusion matrix - SBERT + LogReg", "cm_SBERT_LogReg.png")

    base = pd.read_csv(C.RES_DIR / "baseline_results.csv")
    allres = pd.concat([base, pd.DataFrame([m])[base.columns]], ignore_index=True).round(4)
    allres.to_csv(C.RES_DIR / "final_comparison.csv", index=False)
    print(allres.to_string(index=False))
    print("\nNow write in your report whichever is better ACCORDING TO THESE NUMBERS.")


if __name__ == "__main__":
    main()
