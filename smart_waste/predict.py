import argparse
import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from smart_waste.config import IMAGE_SIZE, LABELS_PATH, MODEL_PATH


def load_predictor(model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
    model = tf.keras.models.load_model(model_path)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return model, labels


def preprocess_image(image_input: str | Path | bytes) -> np.ndarray:
    if isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    else:
        image = Image.open(image_input)

    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def predict_image(model, labels: list[str], image_input: str | Path | bytes, threshold: float = 0.40) -> dict:
    batch = preprocess_image(image_input)
    prediction = model.predict(batch, verbose=0)[0]

    probabilities = {label: float(score) for label, score in zip(labels, prediction)}
    top_index = int(np.argmax(prediction))
    top_label = labels[top_index]
    top_confidence = float(prediction[top_index])

    glass_score = probabilities.get("broken_glass", 0.0)

    return {
        "label": top_label,
        "confidence": top_confidence,
        "probabilities": probabilities,
        "risk_alert": glass_score >= threshold,
        "broken_glass_score": glass_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, labels = load_predictor()
    result = predict_image(model, labels, args.image, threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
