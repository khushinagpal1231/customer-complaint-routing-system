"""Step 4: LDA topic modelling inside each department ("what are customers complaining about?").
Run: python -m src.topics
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from . import config as C
from .utils import load_split

N_TOPICS, N_WORDS = 4, 10


def main():
    train, _ = load_split()
    rows = []
    for dept, g in train.groupby("product"):
        cv = CountVectorizer(max_features=5000, max_df=0.5, min_df=3)
        X = cv.fit_transform(g["text_clean"])
        lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=C.SEED, learning_method="batch", max_iter=15).fit(X)
        names = cv.get_feature_names_out()
        share = lda.transform(X).mean(axis=0)
        for k, comp in enumerate(lda.components_):
            words = ", ".join(names[comp.argsort()[::-1][:N_WORDS]])
            rows.append({"department": dept, "topic": k + 1, "share": round(float(share[k]), 3), "top_words": words})
        print("done", dept)
    pd.DataFrame(rows).to_csv(C.RES_DIR / "lda_topics.csv", index=False)
    print("Saved outputs/results/lda_topics.csv  -> read the words and give each topic a human name in your report.")


if __name__ == "__main__":
    main()
