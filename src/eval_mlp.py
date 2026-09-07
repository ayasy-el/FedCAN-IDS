import json
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

matplotlib.use("Agg")  # headless-safe: simpan figure, tidak membutuhkan display
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns

from data.mlp_dataset import MLPCANDataset
from utils.metrics import MacroF1Score
from utils.mlflow_utils import load_run_id
from utils.params import load_params

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)


# ==========================================================
# Load params.yaml
# ==========================================================

dataset_params = load_params("dataset")
model_params = load_params("model.mlp")
training_params = load_params("training.mlp")
mlflow_params = load_params("mlflow")


# ==========================================================
# Configuration
# ==========================================================

TEST_PATH = f"{dataset_params['featured_dir']}/test.parquet"
TRAIN_PATH = f"{dataset_params['featured_dir']}/train.parquet"
MODEL_PATH = "checkpoints/mlp_best.keras"
NORMALIZE_STATS_PATH = "checkpoints/mlp_norm_stats.json"
RUN_ID_PATH = "checkpoints/mlp_mlflow_run_id.txt"
CLASS_NAMES = ["Normal", "Flooding", "Fuzzing", "Spoofing", "Replay"]
SAMPLE_SIZE = 10_000
SAMPLE_SEED = 42

# Semua output evaluasi disimpan agar hasil antar-run dapat dibandingkan.
REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_JSON_PATH = REPORTS_METRICS_DIR / "mlp_classification_report.json"
CONFUSION_MATRIX_PNG_PATH = REPORTS_FIGURES_DIR / "mlp_confusion_matrix.png"
CONFUSION_MATRIX_NORM_PNG_PATH = (
    REPORTS_FIGURES_DIR / "mlp_confusion_matrix_normalized.png"
)
TRAIN_SAMPLE_CM_PERCENT_PNG_PATH = (
    REPORTS_FIGURES_DIR / "mlp_train_sample_confusion_matrix_percent.png"
)
EVAL_SAMPLE_CM_PERCENT_PNG_PATH = (
    REPORTS_FIGURES_DIR / "mlp_eval_sample_confusion_matrix_percent.png"
)

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_mlp"])


# ==========================================================
# Dataset
# ==========================================================

train_dataset = MLPCANDataset(
    TRAIN_PATH,
    normalize_stats_path=NORMALIZE_STATS_PATH,
    fit_normalize_stats=False,  # gunakan statistik train yang sudah dibuat saat training
    can_id_bits=model_params["can_id_bits"],
    batch_size=training_params["batch_size"],
    shuffle=False,
)
test_dataset = MLPCANDataset(
    TEST_PATH,
    normalize_stats_path=NORMALIZE_STATS_PATH,
    fit_normalize_stats=False,  # gunakan statistik train, jangan hitung ulang
    can_id_bits=model_params["can_id_bits"],
    batch_size=training_params["batch_size"],
    shuffle=False,
)
test_tf = test_dataset.to_tf_dataset()


# ==========================================================
# Load model
# ==========================================================

model = keras.models.load_model(MODEL_PATH, compile=False)
model.summary()
model.compile(
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", MacroF1Score(model_params["num_classes"])],
)
print(f"\nModel loaded successfully: {MODEL_PATH}")


# ==========================================================
# Compile and evaluate
# ==========================================================

keras_results = model.evaluate(test_tf, verbose=1, return_dict=True)

print("\nKeras evaluation:")
for name, value in keras_results.items():
    print(f"{name}: {value:.4f}")


# ==========================================================
# Predict
# ==========================================================

predictions = model.predict(test_tf, verbose=1)
y_true = test_dataset.y
y_pred = np.argmax(np.asarray(predictions), axis=-1)


def stratified_sample_indices(y, max_samples, seed):
    """Ambil sampel tanpa mengubah distribusi kelas secara material."""
    if len(y) <= max_samples:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_counts = counts * max_samples / len(y)
    sample_counts = np.floor(target_counts).astype(int)

    # Bagikan sisa slot ke kelas dengan pecahan terbesar agar total tepat.
    remainder = max_samples - int(sample_counts.sum())
    order = np.argsort(-(target_counts - sample_counts))
    sample_counts[order[:remainder]] += 1

    selected = []
    for cls, n in zip(classes, sample_counts):
        class_indices = np.flatnonzero(y == cls)
        selected.append(rng.choice(class_indices, size=n, replace=False))

    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return selected


def sample_confusion_matrix(dataset, seed):
    indices = stratified_sample_indices(dataset.y, SAMPLE_SIZE, seed)
    sample_y_true = dataset.y[indices]
    sample_y_pred = np.argmax(
        np.asarray(model.predict(dataset.x[indices], verbose=0)), axis=-1
    )
    sample_cm = confusion_matrix(
        sample_y_true,
        sample_y_pred,
        labels=np.arange(len(CLASS_NAMES)),
    )
    sample_cm_percent = (
        sample_cm.astype(float)
        / np.maximum(sample_cm.sum(axis=1, keepdims=True), 1)
        * 100
    )
    return indices, sample_cm, sample_cm_percent


train_sample_indices, train_sample_cm, train_sample_cm_percent = (
    sample_confusion_matrix(train_dataset, SAMPLE_SEED)
)
eval_sample_indices, eval_sample_cm, eval_sample_cm_percent = (
    sample_confusion_matrix(test_dataset, SAMPLE_SEED + 1)
)


def save_sample_confusion_matrix(cm_percent, path, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues",
        vmin=0,
        vmax=100,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


save_sample_confusion_matrix(
    train_sample_cm_percent,
    TRAIN_SAMPLE_CM_PERCENT_PNG_PATH,
    "MLP Train Sample Confusion Matrix (%)",
)
save_sample_confusion_matrix(
    eval_sample_cm_percent,
    EVAL_SAMPLE_CM_PERCENT_PNG_PATH,
    "MLP Eval Sample Confusion Matrix (%)",
)
print(
    f"Train sample confusion matrix disimpan ke: {TRAIN_SAMPLE_CM_PERCENT_PNG_PATH}"
)
print(
    f"Eval sample confusion matrix disimpan ke: {EVAL_SAMPLE_CM_PERCENT_PNG_PATH}"
)


# ==========================================================
# Metrics and classification report
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

report_text = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
)
print("\nClassification Report\n")
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
# Confusion matrices
# ==========================================================

cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
cm_percent = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1) * 100
print("\nConfusion Matrix")
print(cm)

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
plt.title("MLP CAN IDS Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PNG_PATH, dpi=300)
plt.close()
print(f"\nConfusion matrix figure disimpan ke: {CONFUSION_MATRIX_PNG_PATH}")

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
plt.title("MLP CAN IDS Confusion Matrix (Normalized %)")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_NORM_PNG_PATH, dpi=300)
plt.close()
print(f"Normalized confusion matrix disimpan ke: {CONFUSION_MATRIX_NORM_PNG_PATH}")


# ==========================================================
# Save report
# ==========================================================

full_report = {
    "model": "mlp",
    "test_path": TEST_PATH,
    "model_path": MODEL_PATH,
    "n_test_frames": int(len(y_true)),
    "keras_loss": float(keras_results["loss"]),
    "keras_accuracy": float(keras_results["accuracy"]),
    "keras_macro_f1": float(keras_results["macro_f1"]),
    "accuracy": float(accuracy),
    "precision_macro": float(precision_macro),
    "recall_macro": float(recall_macro),
    "f1_macro": float(f1_macro),
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_normalized_percent": cm_percent.tolist(),
    "sample_size_limit": SAMPLE_SIZE,
    "sample_seed": SAMPLE_SEED,
    "train_sample_size": int(len(train_sample_indices)),
    "eval_sample_size": int(len(eval_sample_indices)),
    "train_sample_confusion_matrix": train_sample_cm.tolist(),
    "train_sample_confusion_matrix_percent": train_sample_cm_percent.tolist(),
    "eval_sample_confusion_matrix": eval_sample_cm.tolist(),
    "eval_sample_confusion_matrix_percent": eval_sample_cm_percent.tolist(),
    "class_names": CLASS_NAMES,
}
with open(METRICS_JSON_PATH, "w") as file:
    json.dump(full_report, file, indent=2)
print(f"Classification report disimpan ke: {METRICS_JSON_PATH}")


# ==========================================================
# Log to MLflow (resume the training run)
# ==========================================================

existing_run_id = load_run_id(RUN_ID_PATH)
with mlflow.start_run(run_id=existing_run_id) as run:
    if existing_run_id is None:
        print(
            f"WARNING: {RUN_ID_PATH} tidak ditemukan -- evaluasi dicatat "
            "pada MLflow run baru."
        )
    mlflow.log_metrics(
        {
            "test_keras_loss": float(keras_results["loss"]),
            "test_keras_accuracy": float(keras_results["accuracy"]),
            "test_keras_macro_f1": float(keras_results["macro_f1"]),
            "test_accuracy": float(accuracy),
            "test_precision_macro": float(precision_macro),
            "test_recall_macro": float(recall_macro),
            "test_f1_macro": float(f1_macro),
        }
    )
    for class_name in CLASS_NAMES:
        class_metrics = report_dict[class_name]
        mlflow.log_metrics(
            {
                f"test_precision_{class_name}": class_metrics["precision"],
                f"test_recall_{class_name}": class_metrics["recall"],
                f"test_f1_{class_name}": class_metrics["f1-score"],
            }
        )
    mlflow.log_artifact(str(METRICS_JSON_PATH))
    mlflow.log_artifact(str(CONFUSION_MATRIX_PNG_PATH))
    mlflow.log_artifact(str(CONFUSION_MATRIX_NORM_PNG_PATH))
    mlflow.log_artifact(str(TRAIN_SAMPLE_CM_PERCENT_PNG_PATH))
    mlflow.log_artifact(str(EVAL_SAMPLE_CM_PERCENT_PNG_PATH))

print(f"Metrik evaluasi di-log ke MLflow run: {run.info.run_id}")
