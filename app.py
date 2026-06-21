"""Interface web da demo AI Smart Waste.

Este arquivo usa Streamlit para criar uma tela simples onde o usuario envia
uma imagem e recebe a decisao de seguranca do modelo.
"""

from pathlib import Path

import streamlit as st

from smart_waste.predict import load_predictor, predict_image


# Caminhos dos artefatos gerados por smart_waste.train.
MODEL_PATH = Path("artifacts/model.keras")
LABELS_PATH = Path("artifacts/labels.json")


# Configuracao basica da pagina.
st.set_page_config(page_title="AI Smart Waste", layout="centered")
st.title("AI Smart Waste Glass Detector")
st.caption("Deteccao de risco por vidro quebrado em residuos.")

# A demo depende do modelo treinado. Se ele nao existir, mostramos uma
# mensagem clara em vez de quebrar a aplicacao com erro tecnico.
if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(
        "Modelo nao encontrado. Treine primeiro com: "
        "`python -m smart_waste.train --data-dir data/processed --epochs 10`"
    )
    st.stop()

# Slider para explicar o tradeoff da solucao:
# - limiar baixo: mais sensivel a vidro quebrado, menos falso negativo;
# - limiar alto: menos alarmes, mas pode deixar vidro passar.
threshold = st.slider(
    "Limiar para alerta de vidro quebrado",
    min_value=0.10,
    max_value=0.95,
    value=0.30,
    step=0.05,
)

# Upload da imagem que sera enviada ao modelo.
uploaded_file = st.file_uploader("Envie uma imagem do residuo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Le os bytes da imagem enviada e mostra uma previa para o usuario.
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Imagem analisada", use_container_width=True)

    # Carrega modelo/labels e executa inferencia.
    model, labels = load_predictor(MODEL_PATH, LABELS_PATH)
    result = predict_image(model, labels, image_bytes, threshold=threshold)

    # A decisao principal da demo e de seguranca: alertar ou liberar.
    if result["risk_alert"]:
        st.metric("Decisao de seguranca", "ALERTA")
        st.metric("Score de vidro quebrado", f"{result['broken_glass_score']:.1%}")
        st.error("ALERTA: possivel vidro quebrado detectado.")
    else:
        st.metric("Decisao de seguranca", "SEGURO")
        st.metric("Confianca", f"{result['confidence']:.1%}")
        st.success("Nenhum vidro quebrado detectado acima do limiar.")

    # Classe bruta do modelo e mostrada separadamente para transparencia.
    st.caption(f"Classe bruta do modelo: {result['label']} ({result['confidence']:.1%})")

    # Grafico simples com as probabilidades de cada classe.
    st.write("Probabilidades por classe")
    st.bar_chart(result["probabilities"])
