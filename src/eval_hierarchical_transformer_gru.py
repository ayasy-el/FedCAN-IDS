"""Evaluate the Hierarchical Spatio-Temporal CAN-Aware Transformer-GRU model."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path
from time import perf_counter

import dagshub
import mlflow
import numpy as np
from tensorflow import keras

from data.transformer_gru_dataset import (
    TransformerGRUDataset,
)
from model.hierarchical_transformer_gru import (
    AttentionPooling1D,
    CLSToken,
    ExtractCLS,
    GatedFusion,
    NibbleFrameTokenizer,
    build_hierarchical_transformer_gru,
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
task = validate_experiment(params, "hierarchical_transformer_gru")
num_classes = task["num_classes"]
split = params["split"]
model_params = load_params("model.hierarchical_transformer_gru")
training_params = load_params("training.hierarchical_transformer_gru")
mlflow_params = load_params("mlflow")


# ==========================================================
# Configuration
# ==========================================================

window_size = int(split.get("window_size", 20))
TRAIN_PATH = f"{dataset_params['processed_dir']}/train.parquet"
TEST_PATH = f"{dataset_params['processed_dir']}/test.parquet"
VAL_PATH = f"{dataset_params['processed_dir']}/val.parquet"
MODEL_PATH = "checkpoints/hierarchical_transformer_gru_best.keras"
BEST_TRAINING_METRICS_PATH = (
    "reports/metrics/hierarchical_transformer_gru_best_training_metrics.json"
)
RUN_ID_PATH = "checkpoints/hierarchical_transformer_gru_mlflow_run_id.txt"
CLASS_NAMES = task["class_names"]

REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON_PATH = (
    REPORTS_METRICS_DIR / "hierarchical_transformer_gru_classification_report.json"
)
METRICS_TEXT_PATH = (
    REPORTS_METRICS_DIR / "hierarchical_transformer_gru_classification_report.txt"
)
TRAIN_CONFUSION_MATRIX_PATH = (
    REPORTS_FIGURES_DIR / "hierarchical_transformer_gru_train_confusion_matrix.png"
)
VAL_CONFUSION_MATRIX_PATH = (
    REPORTS_FIGURES_DIR / "hierarchical_transformer_gru_val_confusion_matrix.png"
)
TEST_CONFUSION_MATRIX_PATH = (
    REPORTS_FIGURES_DIR / "hierarchical_transformer_gru_test_confusion_matrix.png"
)
TEST_CONFUSION_MATRIX_COUNTS_PATH = (
    REPORTS_FIGURES_DIR / "hierarchical_transformer_gru_test_confusion_matrix_counts.png"
)

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(
    mlflow_params.get(
        "experiment_hierarchical_transformer_gru",
        "can_ids_hierarchical_transformer_gru",
    )
)


# ==========================================================
# Dataset
# ==========================================================

train_dataset = TransformerGRUDataset(
    TRAIN_PATH,
    window_size=window_size,
    rolling_w=model_params.get("rolling_w", 50),
    num_features=model_params.get("num_features", 5),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split.get("random_seed", 42),
    source_path=dataset_params["prepared_path"],
    label_rule=model_params.get("label_rule", "last_frame"),
)
val_dataset = TransformerGRUDataset(
    VAL_PATH,
    window_size=window_size,
    rolling_w=model_params.get("rolling_w", 50),
    num_features=model_params.get("num_features", 5),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split.get("random_seed", 42),
    source_path=dataset_params["prepared_path"],
    label_rule=model_params.get("label_rule", "last_frame"),
)
test_dataset = TransformerGRUDataset(
    TEST_PATH,
    window_size=window_size,
    rolling_w=model_params.get("rolling_w", 50),
    num_features=model_params.get("num_features", 5),
    batch_size=training_params["batch_size"],
    shuffle=False,
    random_seed=split.get("random_seed", 42),
    source_path=dataset_params["prepared_path"],
    label_rule=model_params.get("label_rule", "last_frame"),
)


# ==========================================================
# Load model
# ==========================================================

custom_objects = {
    "CLSToken": CLSToken,
    "ExtractCLS": ExtractCLS,
    "NibbleFrameTokenizer": NibbleFrameTokenizer,
    "AttentionPooling1D": AttentionPooling1D,
    "GatedFusion": GatedFusion,
}

try:
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False,
    )
except Exception:
    model = build_hierarchical_transformer_gru(
        window_size=window_size,
        num_features=model_params.get("num_features", 5),
        num_classes=num_classes,
        d_model=model_params.get("d_model", 16),
        num_heads=model_params.get("num_heads", 2),
        dim_feedforward=model_params.get("dim_feedforward", 32),
        dropout=model_params.get("dropout", 0.2),
        gru_hidden_size=model_params.get("gru_hidden_size", 32),
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
# Inference Latency Benchmark
# ==========================================================

latency_batch_size = 1
latency_samples = min(100, len(test_dataset.y))
latency_durations = []
if latency_samples > 0:
    for idx in range(latency_samples):
        tok, feat = test_dataset._window_at(idx)
        tok_b = np.expand_dims(tok, 0)
        feat_b = np.expand_dims(feat, 0)
        t0 = perf_counter()
        _ = model([tok_b, feat_b], training=False)
        latency_durations.append((perf_counter() - t0) * 1000.0)  # ms
    mean_latency_ms = float(np.mean(latency_durations[5:]))  # warm-up skipped
    print(f"\nSingle-sample inference latency: {mean_latency_ms:.4f} ms/sample")
else:
    mean_latency_ms = None


# ==========================================================
# Evaluate & Predict
# ==========================================================

test_tf = test_dataset.to_tf_dataset()
test_results = model.evaluate(test_tf, verbose=1, return_dict=True)

print("\nKeras evaluation on Test Set:")
for name, value in test_results.items():
    print(f"  {name}: {value:.4f}")

predictions = model.predict(test_tf, verbose=1)
y_true = test_dataset.y
y_pred = np.argmax(np.asarray(predictions), axis=-1)

test_confusion_matrix_counts, test_confusion_matrix = (
    calculate_confusion_matrices(y_true, y_pred, len(CLASS_NAMES))
)

save_confusion_matrix_figure(
    test_confusion_matrix,
    TEST_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "Hierarchical Transformer-GRU Test Confusion Matrix",
    percentage=True,
)
save_confusion_matrix_figure(
    test_confusion_matrix_counts,
    TEST_CONFUSION_MATRIX_COUNTS_PATH,
    CLASS_NAMES,
    "Hierarchical Transformer-GRU Test Confusion Matrix (Counts)",
    percentage=False,
)

SAMPLE_SIZE = None
SAMPLE_SEED = 42

train_sample_indices = stratified_sample_indices(
    train_dataset.y, SAMPLE_SIZE, SAMPLE_SEED
)
train_sample_predictions = np.argmax(
    model.predict(
        train_dataset.to_tf_dataset(train_sample_indices), verbose=1
    ),
    axis=-1,
)
if len(val_dataset.y):
    val_sample_indices = stratified_sample_indices(
        val_dataset.y, SAMPLE_SIZE, SAMPLE_SEED + 1
    )
    val_sample_predictions = np.argmax(
        model.predict(
            val_dataset.to_tf_dataset(val_sample_indices), verbose=1
        ),
        axis=-1,
    )
else:
    val_sample_indices = np.empty(0, dtype=np.int64)
    val_sample_predictions = np.empty(0, dtype=np.int64)

train_confusion_matrix_counts, train_confusion_matrix = (
    calculate_confusion_matrices(
        train_dataset.y[train_sample_indices],
        train_sample_predictions,
        len(CLASS_NAMES),
    )
)
val_confusion_matrix_counts, val_confusion_matrix = (
    calculate_confusion_matrices(
        val_dataset.y[val_sample_indices],
        val_sample_predictions,
        len(CLASS_NAMES),
    )
)
save_confusion_matrix_figure(
    train_confusion_matrix,
    TRAIN_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "Hierarchical Transformer-GRU Train Confusion Matrix",
    percentage=True,
)
save_confusion_matrix_figure(
    val_confusion_matrix,
    VAL_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "Hierarchical Transformer-GRU Validation Confusion Matrix",
    percentage=True,
)


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

param_counts = parameter_counts(model)
best_training_metrics = load_best_training_metrics(BEST_TRAINING_METRICS_PATH)

full_report = build_evaluation_report(
    model_name="hierarchical_transformer_gru",
    test_results=test_results,
    class_names=CLASS_NAMES,
    best_training_report=best_training_metrics,
    test_classification_report=test_metrics["classification_report"],
    extra={
        "dataset": dataset_params,
        "split": split,
        "model": model_params,
        "training": training_params,
        "parameter_counts": param_counts,
        "mean_latency_ms": mean_latency_ms,
    },
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
if mean_latency_ms is not None:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("inference_latency_ms", mean_latency_ms)

print(f"Metrik evaluasi di-log ke MLflow run: {run_id}")
print("\nEvaluation successfully finished and logged.")
