import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)
from . import config as C


def load_split():
    train, test = pd.read_csv(C.TRAIN_CSV), pd.read_csv(C.TEST_CSV)
    return train, test


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p, "recall_macro": r, "f1_macro": f}


def plot_cm(y_true, y_pred, labels, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=[l[:22] for l in labels]).plot(ax=ax, xticks_rotation=40, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / fname, dpi=130); plt.close()
