from pathlib import Path


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"
HISTORY_PATH = ARTIFACTS_DIR / "history.csv"
