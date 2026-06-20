import json
from pathlib import Path

import tensorflow as tf

from smart_waste.config import BATCH_SIZE, IMAGE_SIZE, SEED


def load_split(data_dir: Path, split: str, shuffle: bool) -> tf.data.Dataset:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="categorical",
    )


def optimize(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.cache().prefetch(tf.data.AUTOTUNE)


def save_labels(class_names: list[str], labels_path: Path) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def load_labels(labels_path: Path) -> list[str]:
    return json.loads(labels_path.read_text(encoding="utf-8"))
