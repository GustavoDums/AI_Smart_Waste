"""Script de treinamento do classificador AI Smart Waste.

Fluxo geral:
1. Carrega imagens de treino e validacao.
2. Cria o modelo definido em model.py.
3. Treina a rede neural.
4. Salva modelo, labels e historico de metricas.
"""

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from smart_waste.config import ARTIFACTS_DIR, HISTORY_PATH, LABELS_PATH, MODEL_PATH
from smart_waste.data import load_split, optimize, save_labels
from smart_waste.model import build_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Le argumentos enviados pelo terminal.
    Exemplo:
        python -m smart_waste.train --data-dir data/processed --epochs 10
    """
    parser = argparse.ArgumentParser(description="Train the Smart Waste image classifier.")
    # Pasta que contem train/ val/ test/.
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    # Quantas vezes o modelo passa pelo dataset de treino inteiro.
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def compute_class_weights(data_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Calcula pesos para compensar classes desbalanceadas.

    Quando uma classe tem muito mais imagens que outra, o modelo pode aprender
    a favorecer a classe majoritaria. O class_weight aumenta a importancia dos
    erros na classe menor durante o treino.
    """
    counts = []
    for class_name in class_names:
        class_dir = data_dir / "train" / class_name
        count = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts.append(count)

    total = sum(counts)
    num_classes = len(class_names)
    return {
        index: total / (num_classes * count)
        for index, count in enumerate(counts)
        if count > 0
    }


def main() -> None:
    """Executa o treino de ponta a ponta."""
    args = parse_args()

    # Garante que a pasta de artefatos existe antes de salvar modelo/labels.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Carrega os dados. Treino embaralha; validacao nao precisa embaralhar.
    train_ds = load_split(args.data_dir, "train", shuffle=True)
    val_ds = load_split(args.data_dir, "val", shuffle=False)

    # O Keras descobre as classes a partir das subpastas.
    # Salvamos essa ordem para usar depois em predict.py e evaluate.py.
    class_names = train_ds.class_names
    save_labels(class_names, LABELS_PATH)
    class_weights = compute_class_weights(args.data_dir, class_names)

    # Cria a arquitetura da rede neural.
    model = build_model(num_classes=len(class_names))

    callbacks = [
        # EarlyStopping para o treino se a validacao parar de melhorar.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        # ModelCheckpoint salva automaticamente a melhor versao do modelo.
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
        class_weight=class_weights,
    )

    # Salva o historico de metricas por epoca para analise posterior.
    pd.DataFrame(history.history).to_csv(HISTORY_PATH, index=False)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")
    print(f"History saved to {HISTORY_PATH}")

if __name__ == "__main__":
    # Permite rodar este arquivo pelo terminal com:
    # python -m smart_waste.train
    main()
