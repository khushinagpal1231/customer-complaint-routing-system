"""Step 1: load CFPB csv -> clean -> balance -> train/test split -> EDA charts.
Run:  python -m src.data_prep            (optional: --products "Mortgage" "Debt collection" ...)
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from . import config as C
from .preprocess import light_clean, full_clean


def savefig(name):
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / name, dpi=130)
    plt.close()


def main(products=None, csv=None):
    csv = csv or C.RAW_CSV
    print(f"Loading {csv} ...")
    df = pd.read_csv(csv, usecols=lambda c: c in {C.COL_TEXT, C.COL_PRODUCT, C.COL_ISSUE, C.COL_DATE})
    df = df.rename(columns={C.COL_TEXT: "text", C.COL_PRODUCT: "product", C.COL_ISSUE: "issue", C.COL_DATE: "date"})
    print("Raw rows:", len(df))

    # ---- inspect + clean ----
    n0 = len(df)
    df = df.dropna(subset=["text", "product"])
    df = df.drop_duplicates(subset="text")
    print(f"After dropping missing/duplicate narratives: {len(df)} (removed {n0 - len(df)})")

    # ---- EDA: full distribution before sampling ----
    df["n_words"] = df["text"].str.split().str.len()
    df["product"].value_counts().head(12).sort_values().plot.barh(figsize=(8, 5), color="#3b82f6")
    plt.title("Top 12 Product categories (before sampling)"); savefig("eda_01_class_distribution.png")

    df["n_words"].clip(upper=800).plot.hist(bins=50, figsize=(8, 4), color="#10b981")
    plt.title("Complaint length (words, clipped at 800)"); plt.xlabel("words"); savefig("eda_02_length.png")

    if "date" in df:
        d = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").value_counts().sort_index()
        d.plot(figsize=(9, 4), marker="o"); plt.title("Complaints per month"); savefig("eda_03_time.png")

    # ---- choose classes ----
    top = df["product"].value_counts()
    keep = products or list(top.head(C.N_CLASSES).index)
    print("Using categories:", keep)
    df = df[df["product"].isin(keep)]

    # ---- balanced sample ----
    n = min(C.N_PER_CLASS, df["product"].value_counts().min())
    df = df.groupby("product", group_keys=False).sample(n=n, random_state=C.SEED).reset_index(drop=True)
    print(f"Balanced sample: {n} per class, total {len(df)}")

    # ---- preprocessing ----
    df["text_light"] = df["text"].map(light_clean)
    df["text_clean"] = df["text"].map(full_clean)
    df = df[df["text_clean"].str.split().str.len() >= 5].reset_index(drop=True)

    # ---- split: test set is locked away here and only used for final evaluation ----
    train, test = train_test_split(df, test_size=C.TEST_SIZE, stratify=df["product"], random_state=C.SEED)
    train.to_csv(C.TRAIN_CSV, index=False)
    test.to_csv(C.TEST_CSV, index=False)
    print(f"Saved train={len(train)} test={len(test)}")

    # sample records for the report
    train.sample(5, random_state=1)[["product", "issue", "text_light"]].to_csv(C.RES_DIR / "sample_records.csv", index=False)

    # before/after example for the preprocessing section
    ex = train.iloc[0]
    (C.RES_DIR / "preprocessing_example.txt").write_text(f"ORIGINAL:\n{ex['text']}\n\nLIGHT:\n{ex['text_light']}\n\nCLEAN:\n{ex['text_clean']}\n")
    print("EDA charts + samples saved in outputs/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+")
    ap.add_argument("--csv")
    a = ap.parse_args()
    main(a.products, a.csv)
