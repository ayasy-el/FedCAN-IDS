from pathlib import Path

import dagshub
import mlflow
import numpy as np
from tensorflow import keras

from data.mlp_dataset import MLPCANDataset
from utils.eval_reporting import (
    build_evaluation_report,
    calculate_classification_metrics,
    calculate_confusion_matrices,
    load_best_training_metrics,
    log_evaluation_to_mlflow,
    save_confusion_matrix_figure,
    save_json_report,
    save_text_report,
    stratified_sample_indices,
)
from utils.metrics import MacroF1Score, MacroPrecision, MacroRecall
from utils.mlflow_utils import load_run_id
from utils.params import load_params
from data.task import validate_experiment

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)


# ==========================================================
# Load params.yaml
# ==========================================================

params = load_params()
dataset_params = params["dataset"]
_, task = validate_experiment(params)
num_classes = task["num_classes"]
model_params = load_params("model.mlp")
training_params = load_params("training.mlp")
mlflow_params = load_params("mlflow")


# ==========================================================
# Configuration
# ==========================================================

TEST_PATH = f"{dataset_params['processed_dir']}/test.parquet"
TRAIN_PATH = f"{dataset_params['processed_dir']}/train.parquet"
MODEL_PATH = "checkpoints/mlp_best.keras"
NORMALIZE_STATS_PATH = "checkpoints/mlp_norm_stats.json"
BEST_TRAINING_METRICS_PATH = "reports/metrics/mlp_best_training_metrics.json"
RUN_ID_PATH = "checkpoints/mlp_mlflow_run_id.txt"
CLASS_NAMES = task["class_names"]
SAMPLE_SIZE = 10_000
SAMPLE_SEED = 42

# Semua output evaluasi disimpan agar hasil antar-run dapat dibandingkan.
REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_JSON_PATH = REPORTS_METRICS_DIR / "mlp_classification_report.json"
METRICS_TEXT_PATH = REPORTS_METRICS_DIR / "mlp_classification_report.txt"
TRAIN_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "mlp_train_confusion_matrix.png"
VAL_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "mlp_val_confusion_matrix.png"
TEST_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "mlp_test_confusion_matrix.png"
TEST_CONFUSION_MATRIX_COUNTS_PATH = (
    REPORTS_FIGURES_DIR / "mlp_test_confusion_matrix_counts.png"
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
val_dataset = MLPCANDataset(
    f"{dataset_params['processed_dir']}/val.parquet",
    normalize_stats_path=NORMALIZE_STATS_PATH,
    fit_normalize_stats=False,  # gunakan statistik train, jangan hitung ulang
    can_id_bits=model_params["can_id_bits"],
    batch_size=training_params["batch_size"],
    shuffle=False,
)
test_dataset = MLPCANDataset(
    TEST_PATH,
    normalize_stats_path=NORMALIZE_STATS_PATH,
    fit_normalize_stats=False,
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
    metrics=[
        "accuracy",
        MacroPrecision(num_classes),
        MacroRecall(num_classes),
        MacroF1Score(num_classes),
    ],
)
print(f"\nModel loaded successfully: {MODEL_PATH}")


# ==========================================================
# Compile and evaluate
# ==========================================================

test_results = model.evaluate(test_tf, verbose=1, return_dict=True)

print("\nKeras evaluation:")
for name, value in test_results.items():
    print(f"{name}: {value:.4f}")


# ==========================================================
# Predict
# ==========================================================

predictions = model.predict(test_tf, verbose=1)
y_true = test_dataset.y
y_pred = np.argmax(np.asarray(predictions), axis=-1)


train_sample_indices = stratified_sample_indices(
    train_dataset.y, SAMPLE_SIZE, SAMPLE_SEED
)
val_sample_indices = stratified_sample_indices(
    val_dataset.y, SAMPLE_SIZE, SAMPLE_SEED + 1
)
train_sample_predictions = np.argmax(
    model.predict(train_dataset.x[train_sample_indices], verbose=1), axis=-1
)
val_sample_predictions = np.argmax(
    model.predict(val_dataset.x[val_sample_indices], verbose=1), axis=-1
)
train_confusion_matrix_counts, train_confusion_matrix = calculate_confusion_matrices(
    train_dataset.y[train_sample_indices],
    train_sample_predictions,
    len(CLASS_NAMES),
)
val_confusion_matrix_counts, val_confusion_matrix = calculate_confusion_matrices(
    val_dataset.y[val_sample_indices],
    val_sample_predictions,
    len(CLASS_NAMES),
)
save_confusion_matrix_figure(
    train_confusion_matrix,
    TRAIN_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "MLP Train Confusion Matrix",
    percentage=True,
)
save_confusion_matrix_figure(
    val_confusion_matrix,
    VAL_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "MLP Validation Confusion Matrix",
    percentage=True,
)
print(f"Train confusion matrix disimpan ke: {TRAIN_CONFUSION_MATRIX_PATH}")
print(f"Validation confusion matrix disimpan ke: {VAL_CONFUSION_MATRIX_PATH}")


# ==========================================================
# Metrics and classification report
# ==========================================================

print("\n======================")
print("Evaluation Result")
print("======================")
test_metrics = calculate_classification_metrics(y_true, y_pred, CLASS_NAMES)
for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
    print(f"{key}: {test_metrics[key]:.4f}")
print("\nClassification Report\n")
print(test_metrics["classification_report_text"])

test_confusion_matrix_counts, test_confusion_matrix = calculate_confusion_matrices(
    y_true, y_pred, len(CLASS_NAMES)
)
print("\nConfusion Matrix")
print(test_confusion_matrix_counts)
save_confusion_matrix_figure(
    test_confusion_matrix_counts,
    TEST_CONFUSION_MATRIX_COUNTS_PATH,
    CLASS_NAMES,
    "MLP Test Confusion Matrix Counts",
    percentage=False,
)
save_confusion_matrix_figure(
    test_confusion_matrix,
    TEST_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "MLP Test Confusion Matrix",
    percentage=True,
)


# ==========================================================
# Save report
# ==========================================================

best_training_report = load_best_training_metrics(BEST_TRAINING_METRICS_PATH)
full_report = build_evaluation_report(
    model_name="mlp",
    test_results=test_results,
    class_names=CLASS_NAMES,
    best_training_report=best_training_report,
    test_classification_report=test_metrics["classification_report"],
)
save_json_report(full_report, METRICS_JSON_PATH)
save_text_report(full_report, METRICS_TEXT_PATH)
print(f"Classification report disimpan ke: {METRICS_JSON_PATH}")
print(f"Text report disimpan ke: {METRICS_TEXT_PATH}")


# ==========================================================
# Log to MLflow (resume the training run)
# ==========================================================

run_id = log_evaluation_to_mlflow(
    run_id=load_run_id(RUN_ID_PATH),
    report=full_report,
    report_path=METRICS_JSON_PATH,
    artifact_paths=[
        TEST_CONFUSION_MATRIX_COUNTS_PATH,
        TEST_CONFUSION_MATRIX_PATH,
        TRAIN_CONFUSION_MATRIX_PATH,
        VAL_CONFUSION_MATRIX_PATH,
        METRICS_TEXT_PATH,
    ],
)
print(f"Metrik evaluasi di-log ke MLflow run: {run_id}")
