"""Evaluate the TCT-IDS model reproducing Gong et al. (2026)."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path
from time import perf_counter

import dagshub
import mlflow
import numpy as np

from data.tct_ids_dataset import TCTIDSDataset
from model.tct_ids import (
    build_tct_ids,
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
model_params = params["model"]["tct_ids"]
training_params = params["training"]["tct_ids"]
mlflow_params = params["mlflow"]


# ==========================================================
# Configuration
# ==========================================================

window_size = int(split["window_size"])
TRAIN_PATH = f"{dataset_params['processed_dir']}/train.parquet"
TEST_PATH = f"{dataset_params['processed_dir']}/test.parquet"
VAL_PATH = f"{dataset_params['processed_dir']}/val.parquet"
MODEL_PATH = "checkpoints/tct_ids_best.keras"
BEST_TRAINING_METRICS_PATH = "reports/metrics/tct_ids_best_training_metrics.json"
RUN_ID_PATH = "checkpoints/tct_ids_mlflow_run_id.txt"
CLASS_NAMES = task["class_names"]

REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON_PATH = REPORTS_METRICS_DIR / "tct_ids_classification_report.json"
METRICS_TEXT_PATH = REPORTS_METRICS_DIR / "tct_ids_classification_report.txt"
TRAIN_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "tct_ids_train_confusion_matrix.png"
VAL_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "tct_ids_val_confusion_matrix.png"
TEST_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "tct_ids_test_confusion_matrix.png"
TEST_CONFUSION_MATRIX_COUNTS_PATH = (
    REPORTS_FIGURES_DIR / "tct_ids_test_confusion_matrix_counts.png"
)

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(
    mlflow_params.get("experiment_tct_ids", "can_ids_tct")
)


# ==========================================================
# Dataset
# ==========================================================

train_dataset = TCTIDSDataset(
    TRAIN_PATH,
    window_size=window_size,
    d_model=model_params.get("d_model", 10),
    alpha=float(model_params.get("alpha_smooth", 1e-7)),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split["random_seed"],
    source_path=dataset_params["prepared_path"],
)
val_dataset = TCTIDSDataset(
    VAL_PATH,
    window_size=window_size,
    d_model=model_params.get("d_model", 10),
    alpha=float(model_params.get("alpha_smooth", 1e-7)),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split["random_seed"],
    source_path=dataset_params["prepared_path"],
)
test_dataset = TCTIDSDataset(
    TEST_PATH,
    window_size=window_size,
    d_model=model_params.get("d_model", 10),
    alpha=float(model_params.get("alpha_smooth", 1e-7)),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split["random_seed"],
    source_path=dataset_params["prepared_path"],
)


# ==========================================================
# Load model
# ==========================================================

model = build_tct_ids(
    window_size=window_size,
    d_model=model_params.get("d_model", 10),
    num_heads=model_params.get("num_heads", 1),
    num_layers=model_params.get("num_layers", 3),
    dim_feedforward=model_params.get("dim_feedforward", 40),
    mlp_hidden_dim=model_params.get("mlp_hidden_dim", 40),
    tcn_filters=model_params.get("tcn_filters", 100),
    tcn_kernel_size=model_params.get("tcn_kernel_size", 2),
    tcn_dilations=model_params.get("tcn_dilations", [1, 2, 4]),
    use_weight_norm=bool(model_params.get("use_weight_norm", True)),
    dropout=model_params.get("dropout", 0.1),
    num_classes=num_classes,
    fusion=model_params.get("fusion", "concat"),
    pooling=model_params.get("pooling", "last"),
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
# Predict & Latency Measurement
# ==========================================================

start = perf_counter()
predictions = model.predict(test_tf, verbose=1)
elapsed = perf_counter() - start
total_windows = max(len(test_dataset.y), 1)
inference_ms_per_window = elapsed * 1000.0 / total_windows
inference_ms_per_batch = (
    elapsed * 1000.0 / max(1, total_windows / training_params["batch_size"])
)
messages_per_second = (total_windows * window_size) / max(elapsed, 1e-6)

print(f"\nLatency: {inference_ms_per_batch:.2f} ms/batch ({training_params['batch_size']} samples)")
print(f"Throughput: {messages_per_second:.1f} messages/sec")

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
    "TCT-IDS Train Confusion Matrix",
    percentage=True,
)
save_confusion_matrix_figure(
    val_matrix,
    VAL_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "TCT-IDS Validation Confusion Matrix",
    percentage=True,
)


# ==========================================================
# Metrics, FNR & classification report
# ==========================================================

test_metrics = calculate_classification_metrics(y_true, y_pred, CLASS_NAMES)

# Calculate FNR per class: FNR_i = FN_i / (TP_i + FN_i) * 100% = (1 - Recall_i) * 100%
fnr_per_class = {}
for i, name in enumerate(CLASS_NAMES):
    tp = test_counts[i, i]
    fn = np.sum(test_counts[i, :]) - tp
    denom = tp + fn
    fnr = (fn / denom * 100.0) if denom > 0 else 0.0
    fnr_per_class[name] = round(fnr, 4)

print("\n--- False Negative Rates (FNR) ---")
for name, fnr_val in fnr_per_class.items():
    print(f"  {name:20s}: {fnr_val:.2f}%")

for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
    print(f"{key}: {test_metrics[key]:.4f}")
print("\nClassification Report\n")
print(test_metrics["classification_report_text"])

save_confusion_matrix_figure(
    test_counts,
    TEST_CONFUSION_MATRIX_COUNTS_PATH,
    CLASS_NAMES,
    "TCT-IDS Test Confusion Matrix Counts",
    percentage=False,
)
save_confusion_matrix_figure(
    test_matrix,
    TEST_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "TCT-IDS Test Confusion Matrix",
    percentage=True,
)


# ==========================================================
# Save report
# ==========================================================

full_report = build_evaluation_report(
    model_name="tct_ids",
    test_results=test_results,
    class_names=CLASS_NAMES,
    best_training_report=load_best_training_metrics(BEST_TRAINING_METRICS_PATH),
    test_classification_report=test_metrics["classification_report"],
    extra={
        "window_size": window_size,
        "stride": int(split["stride"]),
        "parameter_counts": parameter_counts(model),
        "inference_ms_per_window": inference_ms_per_window,
        "inference_ms_per_batch": inference_ms_per_batch,
        "messages_per_second": messages_per_second,
        "fnr_per_class": fnr_per_class,
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
