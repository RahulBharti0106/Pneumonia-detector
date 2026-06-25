# Pneumonia Detection from Chest X-Rays

A deep learning web app that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using transfer learning with MobileNetV2.

Built as a portfolio project demonstrating the full ML pipeline: data preprocessing → model training → evaluation → deployment.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 91.03% |
| Test AUC | 0.9698 |
| Pneumonia Recall | **98.97%** |
| Pneumonia Precision | 88.13% |
| Missed pneumonia cases | 4 out of 390 |

Recall on the PNEUMONIA class was the primary optimization target — in a medical screening context, missing a real case (false negative) is more dangerous than a false alarm.

---

## Model Architecture

- **Base:** MobileNetV2 pretrained on ImageNet (`include_top=False`)
- **Head:** GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)
- **Training:** Two-stage transfer learning
  - Stage 1: Frozen base, head only, lr=1e-3
  - Stage 2: Top 30 layers unfrozen, fine-tune, lr=1e-5
- **Augmentation:** Rotation ±15°, shift 10%, zoom 10%, horizontal flip
- **Class weights:** Applied to handle 3:1 PNEUMONIA/NORMAL imbalance
- **Interpretability:** Grad-CAM heatmaps to visualize model attention

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney on Kaggle.

- 5,216 training images (1,341 NORMAL / 3,875 PNEUMONIA)
- 624 test images
- Validation: 15% split carved from training set (Kaggle's default 16-image val set is too small)

---

## Project Structure

```
pneumonia-detection/
├── models/                          # Not tracked by git — download separately
│   ├── pneumonia_mobilenetv2.keras
│   └── class_indices.json
├── static/
│   ├── main.css
│   └── main.js
├── templates/
│   ├── base.html
│   └── index.html
├── uploads/                         # Runtime temp files, not tracked
├── notebook/
│   └── pneumonia_detection.ipynb    # Full training notebook
├── app.py
├── util.py
├── requirements.txt
└── Procfile
```

---

## Setup & Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model files**

Download `pneumonia_mobilenetv2.keras` and `class_indices.json` from the [Releases](https://github.com/yourusername/pneumonia-detection/releases) page and place them in the `models/` folder.

**4. Run the app**
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## What I'd Improve With More Time

- **Lower the decision threshold** from 0.5 to ~0.3 to further reduce false negatives at the cost of more false alarms — the right trade-off for a screening tool
- **Try EfficientNetB0** — typically 1–2% better AUC on this dataset
- **Larger, more diverse dataset** — this Kaggle dataset comes from a single pediatric hospital; real-world generalization would need data from multiple sources
- **Grad-CAM in the web app** — show the heatmap overlay alongside the prediction so users can see what the model focused on
- **Proper clinical validation** — AUC and recall on a held-out Kaggle test set is not the same as clinical validation; a real deployment would need radiologist review

---

## Tech Stack

- **Model:** TensorFlow / Keras, MobileNetV2
- **Backend:** Flask, Gevent
- **Training environment:** Kaggle (GPU T4)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

> **Disclaimer:** This tool is for educational purposes only. It is not a substitute for professional medical diagnosis.