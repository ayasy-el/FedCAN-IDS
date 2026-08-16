import json
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")  # headless-safe: simpan ke file, tidak butuh display
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import CANDataset
from model.spatial_transformer import SpatialTransformer
from utils.mlflow_utils import load_run_id
from utils.params import load_params

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)


# ==========================
# Configuration
# ==========================

TEST_PATH = "data/processed/car_hacking/test.parquet"

MODEL_PATH = "checkpoints/spatial_best.keras"

BATCH_SIZE = 256

CLASS_NAMES = [
    "Normal",
    "Flooding",
    "Fuzzing",
    "Spoofing",
    "Replay",
]

# Semua output evaluasi (angka & gambar) disimpan ke reports/, bukan
# dicetak/ditampilkan saja -- supaya hasil tiap run bisa dibandingkan
# dan diarsipkan (mis. untuk Leave-One-Group-Out di beberapa kombinasi
# split), bukan cuma tampil sekali di terminal lalu hilang.
REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON_PATH = REPORTS_METRICS_DIR / "spatial_classification_report.json"
CONFUSION_MATRIX_PNG_PATH = REPORTS_FIGURES_DIR / "spatial_confusion_matrix.png"

mlflow_params = load_params("mlflow")
RUN_ID_PATH = "checkpoints/spatial_mlflow_run_id.txt"

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_spatial"])


# ==========================
# Dataset
# ==========================

test_dataset = CANDataset(
    TEST_PATH,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_tf = test_dataset.to_tf_dataset()


# ==========================
# Model
# ==========================

#
# HARUS SAMA DENGAN TRAINING
#
model = SpatialTransformer(
    d_model=4,
    num_heads=2,
    ff_dim=8,
    num_layers=1,
    num_classes=5,
    dropout=0.1,
)


# build model

dummy = {
    "tokens": tf.zeros((1, 10), dtype=tf.int32),
    "token_types": tf.zeros((1, 10), dtype=tf.int32),
    "positions": tf.zeros((1, 10), dtype=tf.int32),
}

model(dummy, training=False)

model.summary()


# ==========================
# Load Weight
# ==========================

model.load_weights(MODEL_PATH)

print("\nModel loaded successfully")


# ==========================
# Prediction
# ==========================

y_true = []
y_pred = []

for x, y in test_tf:
    logits = model(x, training=False)
    pred = tf.argmax(logits, axis=1)

    y_true.extend(y.numpy())
    y_pred.extend(pred.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==========================
# Metrics
# ==========================

print("\n======================")
print("Evaluation Result")
print("======================")

accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("\nAccuracy :", accuracy)
print("Precision Macro :", precision_macro)
print("Recall Macro :", recall_macro)
print("F1 Macro :", f1_macro)


# ==========================
# Classification Report
# ==========================

print("\nClassification Report")

report_text = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)
print(report_text)

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
    output_dict=True,
)


# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(y_true, y_pred)

cm_percent = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100
cm_percent = np.nan_to_num(cm_percent)

print("\nConfusion Matrix")
print(cm)


# ==========================
# Plot Confusion Matrix -> reports/figures/
# ==========================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Spatial Transformer Confusion Matrix")
plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_PNG_PATH, dpi=300)
plt.close()

print(f"\nConfusion matrix figure disimpan ke: {CONFUSION_MATRIX_PNG_PATH}")


# ==========================
# Simpan seluruh hasil evaluasi -> reports/metrics/
# ==========================

full_report = {
    "model": "spatial_transformer",
    "test_path": TEST_PATH,
    "model_path": MODEL_PATH,
    "n_test_samples": len(y_true),
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_normalized_percent": cm_percent.tolist(),
    "class_names": CLASS_NAMES,
}

with open(METRICS_JSON_PATH, "w") as f:
    json.dump(full_report, f, indent=2)

print(f"Classification report (JSON) disimpan ke: {METRICS_JSON_PATH}")


# ==========================
# Log ke MLflow (melanjutkan run training yang sama kalau ada)
# ==========================

_existing_run_id = load_run_id(RUN_ID_PATH)

with mlflow.start_run(run_id=_existing_run_id) as run:
    if _existing_run_id is None:
        print(
            f"PERINGATAN: {RUN_ID_PATH} tidak ditemukan -- log ke MLflow "
            f"run BARU (bukan melanjutkan run training). Jalankan "
            f"train_spatial.py dulu supaya eval ter-link ke run yang sama."
        )

    mlflow.log_metrics(
        {
            "test_accuracy": accuracy,
            "test_precision_macro": precision_macro,
            "test_recall_macro": recall_macro,
            "test_f1_macro": f1_macro,
        }
    )
    for _cname in CLASS_NAMES:
        _m = report_dict[_cname]
        mlflow.log_metrics(
            {
                f"test_precision_{_cname}": _m["precision"],
                f"test_recall_{_cname}": _m["recall"],
                f"test_f1_{_cname}": _m["f1-score"],
            }
        )

    mlflow.log_artifact(str(METRICS_JSON_PATH))
    mlflow.log_artifact(str(CONFUSION_MATRIX_PNG_PATH))

print(f"Metrik evaluasi di-log ke MLflow run: {run.info.run_id}")
