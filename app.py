from pathlib import Path

import streamlit as st

from smart_waste.predict import load_predictor, predict_image


MODEL_PATH = Path("artifacts/model.keras")
LABELS_PATH = Path("artifacts/labels.json")


st.set_page_config(page_title="AI Smart Waste", layout="centered")
st.title("AI Smart Waste")
st.caption("Deteccao de risco por vidro quebrado em residuos.")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(
        "Modelo nao encontrado. Treine primeiro com: "
        "`python -m smart_waste.train --data-dir data/raw --epochs 10`"
    )
    st.stop()

threshold = st.slider(
    "Limiar para alerta de vidro quebrado",
    min_value=0.10,
    max_value=0.95,
    value=0.40,
    step=0.05,
)

uploaded_file = st.file_uploader("Envie uma imagem do residuo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Imagem analisada", use_container_width=True)

    model, labels = load_predictor(MODEL_PATH, LABELS_PATH)
    result = predict_image(model, labels, image_bytes, threshold=threshold)

    st.metric("Classe prevista", result["label"])
    st.metric("Confianca", f"{result['confidence']:.1%}")

    if result["risk_alert"]:
        st.error("ALERTA: possivel vidro quebrado detectado.")
    else:
        st.success("Nenhum vidro quebrado detectado acima do limiar.")

    st.write("Probabilidades por classe")
    st.bar_chart(result["probabilities"])
