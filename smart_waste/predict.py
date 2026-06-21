"""Funcoes de inferencia do projeto.

Inferencia e o uso do modelo ja treinado para analisar uma imagem nova.
Este arquivo e usado tanto pela demo Streamlit (app.py) quanto pelo terminal.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from smart_waste.config import IMAGE_SIZE, LABELS_PATH, MODEL_PATH


def load_predictor(model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
    """Carrega o modelo treinado e os nomes das classes.

    Args:
        model_path: Caminho do arquivo .keras salvo pelo treino.
        labels_path: Caminho do JSON com as classes na ordem correta.

    Returns:
        Uma tupla (model, labels), usada por predict_image().
    """
    # load_model reconstrói a rede neural com os pesos aprendidos.
    model = tf.keras.models.load_model(model_path)

    # labels e uma lista como ["broken_glass", "safe_waste"].
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return model, labels


def preprocess_image(image_input: str | Path | bytes) -> np.ndarray:
    """Prepara uma imagem para entrar no modelo.

    O modelo espera um lote de imagens no formato:
    (quantidade, altura, largura, canais).

    Args:
        image_input: Caminho de imagem ou bytes vindos do upload do Streamlit.

    Returns:
        Um array numpy com shape parecido com (1, 224, 224, 3).
    """
    # Quando a imagem vem do Streamlit, ela chega como bytes.
    if isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    else:
        # Quando vem do terminal, recebemos um caminho de arquivo.
        image = Image.open(image_input)

    # Converte para RGB para garantir 3 canais, mesmo se a imagem for PNG com
    # transparencia ou estiver em escala de cinza.
    image = image.convert("RGB").resize(IMAGE_SIZE)

    # Transforma PIL.Image em array numerico.
    array = tf.keras.utils.img_to_array(image)

    # Adiciona a dimensao do lote. O modelo nao recebe uma imagem "solta"; ele
    # recebe sempre um batch. Aqui o batch tem apenas 1 imagem.
    return np.expand_dims(array, axis=0)


def predict_image(model, labels: list[str], image_input: str | Path | bytes, threshold: float = 0.30) -> dict:
    """Executa predicao em uma imagem e calcula a decisao de seguranca.

    Args:
        model: Modelo Keras carregado.
        labels: Classes na ordem da saida do modelo.
        image_input: Caminho ou bytes da imagem.
        threshold: Limiar de alerta para broken_glass. Um valor menor aumenta
            a chance de alertar vidro quebrado e reduz falsos negativos.

    Returns:
        Dicionario com classe bruta, confianca, probabilidades e alerta.
    """
    # Deixa a imagem no formato que o modelo espera.
    batch = preprocess_image(image_input)

    # model.predict retorna um array com probabilidades por classe.
    # Como usamos batch de 1 imagem, pegamos o primeiro item com [0].
    prediction = model.predict(batch, verbose=0)[0]

    # Cria um dicionario legivel: {"broken_glass": 0.73, "safe_waste": 0.27}
    probabilities = {label: float(score) for label, score in zip(labels, prediction)}

    # Classe de maior probabilidade segundo o softmax.
    top_index = int(np.argmax(prediction))
    top_label = labels[top_index]
    top_confidence = float(prediction[top_index])

    # Para decisao de seguranca, olhamos diretamente o score de broken_glass.
    # Isso permite alertar vidro mesmo se a classe bruta ainda for safe_waste.
    glass_score = probabilities.get("broken_glass", 0.0)
    risk_alert = glass_score >= threshold

    # Este retorno alimenta tanto a interface quanto o terminal.
    return {
        "label": top_label,
        "confidence": top_confidence,
        "probabilities": probabilities,
        "risk_alert": risk_alert,
        "decision": "broken_glass_alert" if risk_alert else "safe_waste",
        "broken_glass_score": glass_score,
    }


def parse_args() -> argparse.Namespace:
    """Le argumentos para predizer uma imagem pelo terminal."""
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    # Caminho da imagem a ser analisada.
    parser.add_argument("image", type=Path)
    # Limiar configuravel para estudar tradeoff entre falso positivo/negativo.
    parser.add_argument("--threshold", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    """Executa inferencia via terminal e imprime JSON com o resultado."""
    args = parse_args()
    model, labels = load_predictor()
    result = predict_image(model, labels, args.image, threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # Permite rodar:
    # python -m smart_waste.predict caminho/da/imagem.jpg
    main()
