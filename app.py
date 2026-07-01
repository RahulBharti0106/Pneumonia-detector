import json
import os

from flask import Flask, jsonify, render_template, request
from gevent.pywsgi import WSGIServer

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

from util import base64_to_pil


app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "pneumonia_mobilenetv2.keras"
)
CLASS_INDICES_PATH = os.path.join(
    os.path.dirname(__file__), "models", "class_indices.json"
)

model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

print(f"Model loaded from {MODEL_PATH}")
print(f"Class mapping: {idx_to_class}")


def model_predict(img):
    """
    Preprocess a PIL image, run inference, and return (label, confidence).
    """
    img = img.convert("RGB").resize((224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred_prob = model.predict(x)[0][0]
    predicted_idx = 1 if pred_prob >= 0.5 else 0
    label = idx_to_class[predicted_idx]
    confidence = pred_prob if predicted_idx == 1 else 1 - pred_prob

    return label, float(confidence)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify(ok=True), 200


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img = base64_to_pil(request.json)
        label, confidence = model_predict(img)
        return jsonify(result=label, confidence=f"{confidence:.1%}")

    return jsonify(error="Invalid request"), 400


if __name__ == "__main__":
    print("Starting server on http://127.0.0.1:5000")
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
