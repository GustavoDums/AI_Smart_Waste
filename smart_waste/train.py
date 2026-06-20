import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from smart_waste.config import ARTIFACTS_DIR, HISTORY_PATH, LABELS_PATH, MODEL_PATH
from smart_waste.data import load_split, optimize, save_labels
from smart_waste.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Smart Waste image classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_split(args.data_dir, "train", shuffle=True)
    val_ds = load_split(args.data_dir, "val", shuffle=False)

    class_names = train_ds.class_names
    save_labels(class_names, LABELS_PATH)

    model = build_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        optimize(train_ds),
        validation_data=optimize(val_ds),
        epochs=args.epochs,
        callbacks=callbacks,
    )

    pd.DataFrame(history.history).to_csv(HISTORY_PATH, index=False)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")
    print(f"History saved to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
