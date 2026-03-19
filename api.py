from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
from pathlib import Path
import numpy as np
import io
from datetime import datetime

app = FastAPI(title="Malaria Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "lenet.keras"

model = load_model(MODEL_PATH)

# Tamaño esperado por el modelo
INPUT_SHAPE = model.input_shape
IMG_HEIGHT = INPUT_SHAPE[1]
IMG_WIDTH = INPUT_SHAPE[2]

# Métricas básicas en memoria
metrics_data = {
    "total_predictions": 0,
    "parasitized_count": 0,
    "uninfected_count": 0,
    "scores": [],
    "last_prediction": None,
    "model_version": "lenet.keras"
}


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def home():
    return {"message": "API de malaria activa"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": metrics_data["model_version"]
    }


@app.get("/metrics")
def metrics():
    avg_score = (
        round(sum(metrics_data["scores"]) / len(metrics_data["scores"]), 2)
        if metrics_data["scores"] else 0.0
    )

    return {
        "total_predictions": metrics_data["total_predictions"],
        "parasitized_count": metrics_data["parasitized_count"],
        "uninfected_count": metrics_data["uninfected_count"],
        "average_score": avg_score,
        "model_version": metrics_data["model_version"]
    }


@app.get("/last_prediction")
def last_prediction():
    if metrics_data["last_prediction"] is None:
        return {"message": "Aún no se ha realizado ninguna predicción"}
    return metrics_data["last_prediction"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        pred = model.predict(img)

        if len(pred.shape) == 2 and pred.shape[1] == 1:
            prob = float(pred[0][0])
            if prob >= 0.5:
                label = "Parasitized"
                score = prob
            else:
                label = "Uninfected"
                score = 1 - prob

        elif len(pred.shape) == 2 and pred.shape[1] == 2:
            class_idx = int(np.argmax(pred[0]))
            score = float(pred[0][class_idx])
            labels = ["Uninfected", "Parasitized"]
            label = labels[class_idx]

        else:
            return {
                "error": "Salida del modelo no esperada",
                "prediction_shape": str(pred.shape),
                "raw_prediction": pred.tolist()
            }

        score_percent = round(score * 100, 2)

        # actualizar métricas
        metrics_data["total_predictions"] += 1
        metrics_data["scores"].append(score_percent)

        if label == "Parasitized":
            metrics_data["parasitized_count"] += 1
        else:
            metrics_data["uninfected_count"] += 1

        metrics_data["last_prediction"] = {
            "label": label,
            "score": score_percent,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return {
            "label": label,
            "score": score_percent,
            "input_shape_model": str(INPUT_SHAPE),
            "prediction_shape": str(pred.shape)
        }

    except Exception as e:
        return {"error": str(e)}