"""Avaliacao do modelo em imagens de teste.

Este script gera os artefatos mais importantes para discutir falsos positivos
e falsos negativos na apresentacao:

- reports/confusion_matrix.png
- reports/classification_report.txt
"""

import argparse
from pathlib import Path

import matplotlib

# Usa backend sem janela grafica. Isso evita erro em ambientes sem Tk/display e
# permite salvar PNG direto em arquivo.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from smart_waste.config import LABELS_PATH, MODEL_PATH, REPORTS_DIR
from smart_waste.data import load_labels, load_split


def parse_args() -> argparse.Namespace:
    """Le argumentos da avaliacao pelo terminal."""
    parser = argparse.ArgumentParser(description="Evaluate the Smart Waste classifier.")
    # Pasta raiz do dataset processado.
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Broken-glass alert threshold. Lower values reduce false negatives.",
    )
    return parser.parse_args()


def main() -> None:
    """Calcula metricas no conjunto de teste e salva relatorios."""
    args = parse_args()

    # Cria a pasta de relatorios se ela ainda nao existir.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Carrega a ordem das classes e o modelo treinado.
    labels = load_labels(LABELS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    # Teste nao deve ser embaralhado; assim a avaliacao fica reprodutivel.
    test_ds = load_split(args.data_dir, "test", shuffle=False)

    # y_true guarda o label real; y_pred guarda o label previsto.
    y_true = []
    y_pred = []

    # Descobre em qual posicao da saida fica a classe broken_glass.
    glass_index = labels.index("broken_glass") if "broken_glass" in labels else None

    for images, targets in test_ds:
        # Prediz um lote de imagens por vez.
        predictions = model.predict(images, verbose=0)

        # targets vem em one-hot; argmax volta para indice da classe.
        y_true.extend(np.argmax(targets.numpy(), axis=1))

        if glass_index is None:
            # Fallback generico: se nao existir broken_glass, usa maior softmax.
            y_pred.extend(np.argmax(predictions, axis=1))
        else:
            # Para seguranca, usamos limiar sobre o score de vidro quebrado.
            # Isso e diferente de sempre escolher a maior probabilidade.
            safe_index = 1 - glass_index
            y_pred.extend(
                glass_index if score >= args.threshold else safe_index
                for score in predictions[:, glass_index]
            )

    # Matriz de confusao mostra acertos e erros por classe.
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusao - limiar {args.threshold:.2f}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Relatorio textual com precision, recall e f1-score.
    report = classification_report(y_true, y_pred, target_names=labels)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    print(report)
    print(f"Confusion matrix saved to {REPORTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    # Permite rodar:
    # python -m smart_waste.evaluate --data-dir data/processed
    main()
