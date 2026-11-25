"""Central settings. Change values here, not inside other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"
RES_DIR = ROOT / "outputs" / "results"
for _d in (DATA_DIR, MODEL_DIR, FIG_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "complaints.csv"          # downloaded CFPB file
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

# CFPB column names
COL_TEXT = "Consumer complaint narrative"
COL_PRODUCT = "Product"
COL_ISSUE = "Issue"
COL_DATE = "Date received"

SEED = 42
N_CLASSES = 5            # how many top Product categories to keep
N_PER_CLASS = 3000       # rows sampled per category (balanced)
TEST_SIZE = 0.15

SBERT_MODEL = "all-MiniLM-L6-v2"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
