import os
import uuid
import json

from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from util import base64_to_pil

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "pneumonia_mobilenetv2.keras"
)
CLASS_INDICES_PATH = os.path.join(
    os.path.dirname(__file__), "models", "class_indices.json"
)

model = load_model(MODEL_PATH)

# Load class mapping: {'NORMAL': 0, 'PNEUMONIA': 1}
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)

# Invert to {0: 'NORMAL', 1: 'PNEUMONIA'}
idx_to_class = {v: k for k, v in class_indices.items()}

print(f"Model loaded from {MODEL_PATH}")
print(f"Class mapping: {idx_to_class}")


# ── Prediction helper ─────────────────────────────────────────────────────────
def model_predict(img_path):
    """
    Load image from path, preprocess exactly as during training,
    run inference, return (label, confidence).
    """
    # Must match training: 224x224, MobileNetV2 preprocess_input
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)  # shape (224, 224, 3)
    x = np.expand_dims(x, axis=0)  # shape (1, 224, 224, 3)
    x = preprocess_input(x)  # scale to [-1, 1]

    pred_prob = model.predict(x)[0][0]  # single sigmoid output

    # pred_prob close to 1 → PNEUMONIA, close to 0 → NORMAL
    predicted_idx = 1 if pred_prob >= 0.5 else 0
    label = idx_to_class[predicted_idx]
    confidence = pred_prob if predicted_idx == 1 else 1 - pred_prob

    return label, float(confidence)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Decode base64 image from request
        img = base64_to_pil(request.json)

        # Use a unique filename per request — avoids overwriting if two
        # users upload at the same time
        filename = f"{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img.save(img_path)

        label, confidence = model_predict(img_path)

        # Clean up the temp file after prediction
        os.remove(img_path)

        return jsonify(result=label, confidence=f"{confidence:.1%}")

    return jsonify(error="Invalid request"), 400


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
