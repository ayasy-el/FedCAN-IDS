"""Evaluate the CAN-AE-Transformer reproduction."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path
from time import perf_counter

import dagshub
import mlflow
import numpy as np

from data.can_ae_transformer_dataset import CANAeTransformerDataset
from model.can_ae_transformer import (
    build_can_ae_transformer,
    parameter_counts,
)
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
from utils.mlflow_utils import load_run_id
from utils.metrics import MacroF1Score, MacroPrecision, MacroRecall
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
split = params["split"]
model_params = params["model"]["can_ae_transformer"]
training_params = params["training"]["can_ae_transformer"]
mlflow_params = params["mlflow"]


# ==========================================================
# Configuration
# ==========================================================

window_size = int(split["window_size"])
TRAIN_PATH = f"{dataset_params['processed_dir']}/train.parquet"
TEST_PATH = f"{dataset_params['processed_dir']}/test.parquet"
VAL_PATH = f"{dataset_params['processed_dir']}/val.parquet"
MODEL_PATH = "checkpoints/can_ae_transformer_best.keras"
BEST_TRAINING_METRICS_PATH = "reports/metrics/can_ae_transformer_best_training_metrics.json"
RUN_ID_PATH = "checkpoints/can_ae_transformer_mlflow_run_id.txt"
CLASS_NAMES = task["class_names"]

REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON_PATH = REPORTS_METRICS_DIR / "can_ae_transformer_classification_report.json"
METRICS_TEXT_PATH = REPORTS_METRICS_DIR / "can_ae_transformer_classification_report.txt"
TRAIN_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "can_ae_transformer_train_confusion_matrix.png"
VAL_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "can_ae_transformer_val_confusion_matrix.png"
TEST_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "can_ae_transformer_test_confusion_matrix.png"
TEST_CONFUSION_MATRIX_COUNTS_PATH = REPORTS_FIGURES_DIR / "can_ae_transformer_test_confusion_matrix_counts.png"

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_can_ae_transformer"])


# ==========================================================
# Dataset
# ==========================================================

train_dataset = CANAeTransformerDataset(
    TRAIN_PATH,
    window_size=window_size,
    d_model=model_params["d_model"],
    granularity=float(model_params.get("granularity", 1e-8)),
    max_time_position=int(model_params.get("max_time_position", 10000)),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split["random_seed"],
    source_path=dataset_params["prepared_path"],
)
val_dataset = CANAeTransformerDataset(
    VAL_PATH,
    window_size=window_size,
    d_model=model_params["d_model"],
    granularity=float(model_params.get("granularity", 1e-8)),
    max_time_position=int(model_params.get("max_time_position", 10000)),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split["random_seed"],
    source_path=dataset_params["prepared_path"],
)
test_dataset = CANAeTransformerDataset(
    TEST_PATH,
    window_size=window_size,
    d_model=model_params["d_model"],
    granularity=float(model_params.get("granularity", 1e-8)),
    max_time_position=int(model_params.get("max_time_position", 10000)),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split["random_seed"],
    source_path=dataset_params["prepared_path"],
)


# ==========================================================
# Load model
# ==========================================================

model = build_can_ae_transformer(
    window_size=window_size,
    d_model=model_params["d_model"],
    num_heads=model_params["num_heads"],
    num_layers=model_params["num_layers"],
    dim_feedforward=model_params["dim_feedforward"],
    dropout=model_params["dropout"],
    num_classes=num_classes,
)
model.load_weights(MODEL_PATH)
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

test_tf = test_dataset.to_tf_dataset()
test_results = model.evaluate(test_tf, verbose=1, return_dict=True)
print("\nKeras evaluation:")
for name, value in test_results.items():
    print(f"{name}: {value:.4f}")


# ==========================================================
# Predict
# ==========================================================

start = perf_counter()
predictions = model.predict(test_tf, verbose=1)
elapsed = perf_counter() - start
inference_ms_per_window = elapsed * 1000.0 / max(len(test_dataset.y), 1)

y_true = test_dataset.y
y_pred = np.argmax(np.asarray(predictions), axis=-1)

train_predictions = np.argmax(
    model.predict(train_dataset.to_tf_dataset(), verbose=1), axis=-1
)
if len(val_dataset.y):
    val_predictions = np.argmax(
        model.predict(val_dataset.to_tf_dataset(), verbose=1), axis=-1
    )
else:
    val_predictions = np.empty(0, dtype=np.int64)

train_counts, train_matrix = calculate_confusion_matrices(
    train_dataset.y, train_predictions, len(CLASS_NAMES)
)
val_counts, val_matrix = calculate_confusion_matrices(
    val_dataset.y, val_predictions, len(CLASS_NAMES)
)
test_counts, test_matrix = calculate_confusion_matrices(
    y_true, y_pred, len(CLASS_NAMES)
)

save_confusion_matrix_figure(
    train_matrix,
    TRAIN_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "CAN-AE-Transformer Train Confusion Matrix",
    percentage=True,
)
save_confusion_matrix_figure(
    val_matrix,
    VAL_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "CAN-AE-Transformer Validation Confusion Matrix",
    percentage=True,
)


# ==========================================================
# Metrics and classification report
# ==========================================================

test_metrics = calculate_classification_metrics(y_true, y_pred, CLASS_NAMES)
for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
    print(f"{key}: {test_metrics[key]:.4f}")
print("\nClassification Report\n")
print(test_metrics["classification_report_text"])

save_confusion_matrix_figure(
    test_counts,
    TEST_CONFUSION_MATRIX_COUNTS_PATH,
    CLASS_NAMES,
    "CAN-AE-Transformer Test Confusion Matrix Counts",
    percentage=False,
)
save_confusion_matrix_figure(
    test_matrix,
    TEST_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "CAN-AE-Transformer Test Confusion Matrix",
    percentage=True,
)


# ==========================================================
# Save report
# ==========================================================

full_report = build_evaluation_report(
    model_name="can_ae_transformer",
    test_results=test_results,
    class_names=CLASS_NAMES,
    best_training_report=load_best_training_metrics(BEST_TRAINING_METRICS_PATH),
    test_classification_report=test_metrics["classification_report"],
    extra={
        "window_size": window_size,
        "stride": int(split["stride"]),
        "parameter_counts": parameter_counts(model),
        "inference_ms_per_window": inference_ms_per_window,
    },
)
save_json_report(full_report, METRICS_JSON_PATH)
save_text_report(full_report, METRICS_TEXT_PATH)


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
