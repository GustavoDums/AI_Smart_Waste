"""Funcoes de leitura e escrita relacionadas ao dataset.

Este modulo e usado por treino e avaliacao. Ele sabe carregar as imagens do
formato de pastas esperado pelo Keras:

data/processed/train/broken_glass
data/processed/train/safe_waste
"""

import json
from pathlib import Path

import tensorflow as tf

from smart_waste.config import BATCH_SIZE, IMAGE_SIZE, SEED


def load_split(data_dir: Path, split: str, shuffle: bool) -> tf.data.Dataset:
    """Carrega uma divisao do dataset: train, val ou test.

    Args:
        data_dir: Pasta raiz do dataset, como data/processed.
        split: Nome da divisao que sera carregada: train, val ou test.
        shuffle: Se True, embaralha as imagens. No treino usamos True; na
            avaliacao/teste usamos False para manter ordem estavel.

    Returns:
        Um tf.data.Dataset com lotes de imagens e labels em formato one-hot.
    """
    # Monta o caminho final, por exemplo: data/processed/train.
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    # image_dataset_from_directory le automaticamente as subpastas como classes.
    # Exemplo: broken_glass e safe_waste viram labels do modelo.
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="categorical",
    )


def optimize(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Melhora a performance do pipeline de dados.

    cache() evita reler/processar as mesmas imagens repetidamente.
    prefetch() prepara o proximo lote enquanto o modelo treina no lote atual.
    """
    return dataset.cache().prefetch(tf.data.AUTOTUNE)


def save_labels(class_names: list[str], labels_path: Path) -> None:
    """Salva a lista de classes na mesma ordem usada pelo modelo.

    Isso e essencial para a inferencia: se o modelo retorna [0.8, 0.2],
    precisamos saber qual classe corresponde a cada posicao.
    """
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def load_labels(labels_path: Path) -> list[str]:
    """Carrega os nomes das classes salvos no treino."""
    return json.loads(labels_path.read_text(encoding="utf-8"))
