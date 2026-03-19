import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Malaria Classifier", layout="centered")
st.title("🦠 Clasificador de Malaria (Parasitized vs Uninfected)")
st.write("Sube una imagen de una célula y la aplicación la enviará a la API para predecir si está parasitada o no.")

uploaded = st.file_uploader("Sube una imagen de célula (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Imagen cargada", use_container_width=True)

    if st.button("Predecir"):
        try:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            files = {
                "file": ("imagen.png", buffer, "image/png")
            }

            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                label = result.get("label", "Sin etiqueta")
                score = result.get("score", None)

                st.success(f"Predicción: {label}")

                if score is not None:
                    st.write(f"Confianza: **{score:.2f}%**")

                st.json(result)

            else:
                st.error(f"Error en la API: {response.status_code}")
                st.text(response.text)

        except requests.exceptions.ConnectionError:
            st.error("No fue posible conectar con la API. Verifica que FastAPI esté corriendo en http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")