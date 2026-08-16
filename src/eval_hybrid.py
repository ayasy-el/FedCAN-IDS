import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib

matplotlib.use("Agg")  # headless-safe: simpan ke file, tidak butuh display
import matplotlib.pyplot as plt
import seaborn as sns

from data.temporal_dataset import (
    TemporalCANDataset,
)

from model.hybrid_ids import HybridIDS


# ==========================================================
# Configuration
# ==========================================================

TEST_PATH = "data/processed/car_hacking/featured/test.parquet"

MODEL_PATH = "checkpoints/hybrid_best.keras"

# Statistik normalisasi (mean/std) HARUS sama persis dengan yang dipakai
# saat training (dihitung dari train, disimpan train_hybrid.py) -- test
# TIDAK BOLEH menghitung statistiknya sendiri, itu data leakage.
NORMALIZE_STATS_PATH = "checkpoints/temporal_norm_stats.json"

SEQ_LEN = 32
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
# split), bukan cuma tampil sekali di terminal lalu hilang. Konsisten
# dengan eval_spatial.py.
REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON_PATH = REPORTS_METRICS_DIR / "hybrid_classification_report.json"
CONFUSION_MATRIX_PNG_PATH = REPORTS_FIGURES_DIR / "hybrid_confusion_matrix.png"
CONFUSION_MATRIX_NORM_PNG_PATH = (
    REPORTS_FIGURES_DIR / "hybrid_confusion_matrix_normalized.png"
)


# ==========================================================
# Dataset
# ==========================================================

test_dataset = TemporalCANDataset(
    TEST_PATH,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    shuffle=False,
    normalize_stats_path=NORMALIZE_STATS_PATH,
    fit_normalize_stats=False,  # muat stats dari train, JANGAN hitung ulang
)

test_tf = test_dataset.to_tf_dataset()


# ==========================================================
# Build model
# ==========================================================

#
# HARUS SAMA DENGAN TRAINING
#
model = HybridIDS(
    d_model=4,
    num_heads=2,
    ff_dim=8,
    num_layers=1,
    gru_units=16,
    num_classes=5,
    dropout=0.1,
)


dummy = {
    "tokens": tf.zeros(
        (1, SEQ_LEN, 10),
        dtype=tf.int32,
    ),
    "token_types": tf.zeros(
        (1, SEQ_LEN, 10),
        dtype=tf.int32,
    ),
    "positions": tf.zeros(
        (1, SEQ_LEN, 10),
        dtype=tf.int32,
    ),
    "temporal_features": tf.zeros(
        (1, SEQ_LEN, 9),
        dtype=tf.float32,
    ),
}

_ = model(dummy)

model.summary()


# ==========================================================
# Load weights
# ==========================================================

model.load_weights(MODEL_PATH)

print(f"\nModel loaded successfully: {MODEL_PATH}")


# ==========================================================
# Compile (dipakai untuk model.evaluate loss/accuracy bawaan Keras)
# ==========================================================

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        "accuracy",
    ],
)


# ==========================================================
# Evaluate (loss & accuracy bawaan Keras)
# ==========================================================

keras_loss, keras_accuracy = model.evaluate(
    test_tf,
    verbose=1,
)

print()
print(f"Test Loss     : {keras_loss:.4f}")
print(f"Test Accuracy : {keras_accuracy:.4f}")


# ==========================================================
# Predict
# ==========================================================

y_true = []
y_pred = []

for x_batch, y_batch in test_tf:
    logits = model(
        x_batch,
        training=False,
    )

    pred = tf.argmax(
        logits,
        axis=-1,
    )

    y_true.extend(y_batch.numpy())
    y_pred.extend(pred.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==========================================================
# Metrics
# ==========================================================

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


# ==========================================================
# Classification Report
# ==========================================================

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


# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_true, y_pred)

cm_percent = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100
cm_percent = np.nan_to_num(cm_percent)

print("\nConfusion Matrix")
print(cm)


# ==========================================================
# Plot Confusion Matrix (raw count) -> reports/figures/
# ==========================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Blues",
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Hybrid IDS Confusion Matrix")
plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_PNG_PATH, dpi=300)
plt.close()

print(f"\nConfusion matrix figure disimpan ke: {CONFUSION_MATRIX_PNG_PATH}")


# ==========================================================
# Plot Confusion Matrix (normalized %) -> reports/figures/
# ==========================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".2f",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Blues",
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Hybrid IDS Confusion Matrix (Normalized %)")
plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_NORM_PNG_PATH, dpi=300)
plt.close()

print(
    f"Confusion matrix (normalized) figure disimpan ke: {CONFUSION_MATRIX_NORM_PNG_PATH}"
)


# ==========================================================
# Simpan seluruh hasil evaluasi -> reports/metrics/
# ==========================================================

full_report = {
    "model": "hybrid_ids",
    "test_path": TEST_PATH,
    "model_path": MODEL_PATH,
    "n_test_samples": int(len(y_true)),
    "keras_loss": float(keras_loss),
    "keras_accuracy": float(keras_accuracy),
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
