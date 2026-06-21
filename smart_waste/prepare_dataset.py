"""Prepara uma divisao balanceada do dataset.

O dataset original pode vir com train/val/test desbalanceados. Este script foi criado apenas 
para juntar todas as imagens por classe e criar uma nova divisao em data/processed, para
garantir que o dataset esteja corretamente balanceado.

- 70% treino
- 15% validacao
- 15% teste

Ele copia arquivos, nao move. Assim data/raw continua preservado.
"""

import argparse
import random
import shutil
from pathlib import Path


# Extensoes de imagem aceitas pelo script.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Proporcao usada para criar as novas divisoes.
SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

def parse_args() -> argparse.Namespace:
    """Le argumentos para preparar o dataset pelo terminal.
    Exemplo:
        python -m smart_waste.prepare_dataset --source-dir data/raw --output-dir data/processed --overwrite
    """
    parser = argparse.ArgumentParser(
        description="Create a balanced train/val/test dataset without changing the original files."
    )
    # Pasta original onde o usuario colocou as imagens.
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))

    # Pasta gerada pelo script, usada no treino.
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    # Semente para embaralhamento reprodutivel.
    parser.add_argument("--seed", type=int, default=42)

    # Permite apagar e recriar data/processed.
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before copying files.",
    )
    return parser.parse_args()


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    """Coleta imagens do dataset original agrupando por classe.

    O script espera uma estrutura parecida com:
        data/raw/train/broken_glass
        data/raw/test/broken_glass
        data/raw/val/safe_waste

    Ele ignora se a imagem veio de train, val ou test original e junta tudo por
    classe para criar uma nova divisao melhor.
    """
    images_by_class: dict[str, list[Path]] = {}

    # Percorre as divisoes originais: train, val, test.
    for split_dir in source_dir.iterdir():
        if not split_dir.is_dir():
            continue
        # Percorre as classes dentro de cada divisao.
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            # setdefault cria a lista da classe se ela ainda nao existir.
            images_by_class.setdefault(class_name, [])

            # Adiciona apenas arquivos com extensao de imagem aceita.
            images_by_class[class_name].extend(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    return images_by_class


def copy_split(class_name: str, images: list[Path], output_dir: Path) -> dict[str, int]:
    """Copia imagens de uma classe para train/val/test.

    Args:
        class_name: Nome da classe, por exemplo broken_glass.
        images: Lista ja embaralhada de imagens daquela classe.
        output_dir: Pasta final, geralmente data/processed.

    Returns:
        Dicionario com a quantidade copiada para cada split.
    """
    total = len(images)

    # Calcula os pontos de corte da lista embaralhada.
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    # Fatia a lista em tres partes.
    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    counts = {}
    for split, split_images in split_map.items():
        # Exemplo de destino: data/processed/train/broken_glass.
        destination = output_dir / split / class_name
        destination.mkdir(parents=True, exist_ok=True)
        counts[split] = len(split_images)

        for index, source in enumerate(split_images, start=1):
            # Renomeia de forma padronizada para evitar nomes repetidos vindos
            # de pastas diferentes do dataset original.
            target = destination / f"{class_name}_{index:05d}{source.suffix.lower()}"

            # copy2 preserva metadados basicos do arquivo.
            shutil.copy2(source, target)

    return counts


def main() -> None:
    """Executa a preparacao completa do dataset."""
    args = parse_args()

    # Falha cedo se a pasta original nao existe.
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {args.source_dir}")

    # Se solicitado, apaga a pasta processada para recriar do zero.
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    # Protecao contra sobrescrever arquivos sem querer.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"{args.output_dir} already exists. Use --overwrite to rebuild it."
        )

    # Define a semente antes de embaralhar as imagens.
    random.seed(args.seed)

    # Junta todas as imagens por classe.
    images_by_class = collect_images(args.source_dir)
    if not images_by_class:
        raise RuntimeError(f"No images found in {args.source_dir}")

    # Para cada classe, embaralha e copia nas proporcoes configuradas.
    for class_name, images in sorted(images_by_class.items()):
        random.shuffle(images)
        counts = copy_split(class_name, images, args.output_dir)
        print(
            f"{class_name}: total={len(images)} "
            f"train={counts['train']} val={counts['val']} test={counts['test']}"
        )


if __name__ == "__main__":
    # Permite rodar:
    # python -m smart_waste.prepare_dataset
    main()
