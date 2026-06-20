import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from smart_waste.config import LABELS_PATH, MODEL_PATH, REPORTS_DIR
from smart_waste.data import load_labels, load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Smart Waste classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_labels(LABELS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    test_ds = load_split(args.data_dir, "test", shuffle=False)

    y_true = []
    y_pred = []

    for images, targets in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(targets.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusao - AI Smart Waste")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    report = classification_report(y_true, y_pred, target_names=labels)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    print(report)
    print(f"Confusion matrix saved to {REPORTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
