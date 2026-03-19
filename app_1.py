import streamlit as st
import streamlit as st
from PIL import Image
from pathlib import Path

from src.config import MODEL_PATH
from src.predict import predict_pil

st.set_page_config(page_title="Malaria Classifier", layout="centered")
st.title("🦠 Clasificador de Malaria (Parasitized vs Uninfected)")

if not Path(MODEL_PATH).exists():
    st.error(
        "No encuentro el modelo en artifacts/lenet.keras.\n\n"
        "Primero entrena y genera el modelo con:\n"
        "python -m src.train"
    )
    st.stop()

uploaded = st.file_uploader("Sube una imagen de célula (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Imagen cargada", use_container_width=True)

    if st.button("Predecir"):
        pred, prob = predict_pil(img)
        label = "Parasitized (P)" if pred == "P" else "Uninfected (U)"
        st.success(f"Predicción: **{label}**")
        st.write(f"Probabilidad (salida sigmoid): `{prob:.4f}`")