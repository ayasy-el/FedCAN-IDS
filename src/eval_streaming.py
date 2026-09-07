from pathlib import Path

import dagshub
import mlflow
import numpy as np
import tensorflow as tf

from data.streaming_dataset import StreamingCANDataset
from model.streaming_ids import StreamingCANIDS
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
from utils.params import load_params

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)


# ==========================================================
# Load params.yaml
# ==========================================================

dataset_params = load_params("dataset")
model_params = load_params("model.streaming")
training_params = load_params("training.streaming")
mlflow_params = load_params("mlflow")


# ==========================================================
# Configuration
# ==========================================================

TEST_PATH = f"{dataset_params['featured_dir']}/test.parquet"
MODEL_PATH = "checkpoints/streaming_best.keras"
NORMALIZE_STATS_PATH = "checkpoints/streaming_norm_stats.json"
RUN_ID_PATH = "checkpoints/streaming_mlflow_run_id.txt"
BEST_TRAINING_METRICS_PATH = "reports/metrics/streaming_best_training_metrics.json"
CLASS_NAMES = ["Normal", "Flooding", "Fuzzing", "Spoofing", "Replay"]

# Semua output evaluasi disimpan agar hasil antar-run dapat dibandingkan.
REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_JSON_PATH = REPORTS_METRICS_DIR / "streaming_classification_report.json"
METRICS_TEXT_PATH = REPORTS_METRICS_DIR / "streaming_classification_report.txt"
TEST_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "streaming_test_confusion_matrix.png"
TEST_CONFUSION_MATRIX_COUNTS_PATH = (
    REPORTS_FIGURES_DIR / "streaming_test_confusion_matrix_counts.png"
)

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_streaming"])


# ==========================================================
# Dataset
# ==========================================================

test_dataset = StreamingCANDataset(
    f"{dataset_params['featured_dir']}/test.parquet",
    chunk_len=training_params["chunk_len"],
    batch_size=training_params["batch_size"],
    shuffle=False,
    normalize_stats_path=NORMALIZE_STATS_PATH,
    fit_normalize_stats=False,  # gunakan statistik train, jangan hitung ulang
)
test_tf = test_dataset.to_tf_dataset()


# ==========================================================
# Build model, load weights, and initialize streaming state
# ==========================================================

model = StreamingCANIDS(**model_params)
model(
    {
        "can_id": tf.zeros(
            (training_params["batch_size"], training_params["chunk_len"]),
            dtype=tf.int32,
        ),
        "numeric": tf.zeros(
            (training_params["batch_size"], training_params["chunk_len"], 11),
            dtype=tf.float32,
        ),
        "stream_id": tf.fill((training_params["batch_size"],), b"build"),
        "valid_mask": tf.ones(
            (training_params["batch_size"], training_params["chunk_len"]),
            dtype=tf.float32,
        ),
    }
)
model.summary()
model.load_weights(MODEL_PATH)
model.initialize_stream_state(training_params["batch_size"])
print(f"\nModel loaded successfully: {MODEL_PATH}")


# ==========================================================
# Compile and evaluate
# ==========================================================

# State KV persistent adalah resource variable. XLA harus dimatikan agar
# resource CPU tidak diakses dari graph GPU yang terkompilasi otomatis.
model.compile(jit_compile=False)
test_results = model.evaluate(
    test_tf,
    steps=test_dataset.num_samples,
    verbose=1,
    return_dict=True,
)

print("\nKeras evaluation:")
for name, value in test_results.items():
    print(f"{name}: {value:.4f}")


# ==========================================================
# Predict
# ==========================================================

model.reset_stream_state()
logits = model.predict(
    test_tf,
    steps=test_dataset.num_samples,
    verbose=1,
)

# Generator deterministic: pass kedua mengambil label dan valid_mask yang
# sama dengan prediction pass, termasuk filtering padding antar-session.
y_true_batches, valid_batches = [], []
for batch, labels in test_dataset.to_tf_dataset():
    y_true_batches.append(labels.numpy())
    valid_batches.append(batch["valid_mask"].numpy().astype(bool))

y_true = np.concatenate(y_true_batches, axis=0)
valid_mask = np.concatenate(valid_batches, axis=0)
y_pred = np.argmax(np.asarray(logits), axis=-1)
y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]


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


# ==========================================================
# Confusion matrices
# ==========================================================

test_confusion_matrix_counts, test_confusion_matrix = calculate_confusion_matrices(
    y_true, y_pred, len(CLASS_NAMES)
)
print("\nConfusion Matrix")
print(test_confusion_matrix_counts)
save_confusion_matrix_figure(
    test_confusion_matrix_counts,
    TEST_CONFUSION_MATRIX_COUNTS_PATH,
    CLASS_NAMES,
    "Streaming Test Confusion Matrix Counts",
    percentage=False,
)
save_confusion_matrix_figure(
    test_confusion_matrix,
    TEST_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "Streaming Test Confusion Matrix",
    percentage=True,
)


# ==========================================================
# Save report
# ==========================================================

full_report = build_evaluation_report(
    model_name="causal_compressed_kv",
    test_results=test_results,
    class_names=CLASS_NAMES,
    best_training_report=load_best_training_metrics(BEST_TRAINING_METRICS_PATH),
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
        METRICS_TEXT_PATH,
    ],
)
print(f"Metrik evaluasi di-log ke MLflow run: {run_id}")
