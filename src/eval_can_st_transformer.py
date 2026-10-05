"""Evaluate the Spatial-Temporal CAN Transformer IDS (Jo & Kim, 2024).

Follows the standard FedCAN-IDS evaluation reporting pipeline:
- Calculates classification metrics and confusion matrices for train, val, and test.
- Adds paper-specific per-attack scenario evaluation (Flooding, Fuzzy, Malfunction).
- Adds paper-specific range expansion analysis (Top-1 to Top-6) and ROC curve.
- Logs all standard metrics and artifacts to MLflow.
"""

from pathlib import Path
from time import perf_counter

import dagshub
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras
import tensorflow as tf

from data.can_st_transformer_dataset import CANSTTransformerDataset
from model.can_st_transformer import build_can_st_transformer
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
from utils.params import load_params
from data.task import validate_experiment

import os

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)



# ==========================================================
# Load params.yaml
# ==========================================================

params = load_params()
dataset_params = params["dataset"]
_, task = validate_experiment(params)
split = params["split"]
model_params = params["model"]["can_st_transformer"]
training_params = params["training"]["can_st_transformer"]
mlflow_params = params["mlflow"]


# ==========================================================
# Configuration
# ==========================================================

window_size = int(model_params.get("window_size", split.get("window_size", 64)))
range_adjustment = int(model_params.get("range_adjustment", 0))

TEST_PATH = f"{dataset_params['processed_dir']}/test.parquet"
TRAIN_PATH = f"{dataset_params['processed_dir']}/train.parquet"
VAL_PATH = f"{dataset_params['processed_dir']}/val.parquet"
MODEL_PATH = "checkpoints/can_st_transformer_best.keras"
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "checkpoints/can_st_transformer_final.keras"

BEST_TRAINING_METRICS_PATH = "reports/metrics/can_st_transformer_best_training_metrics.json"
RUN_ID_PATH = "checkpoints/can_st_transformer_mlflow_run_id.txt"
CLASS_NAMES = ["Normal", "Attack"]
SAMPLE_SIZE = None
SAMPLE_SEED = 42

REPORTS_METRICS_DIR = Path("reports/metrics")
REPORTS_FIGURES_DIR = Path("reports/figures")
REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON_PATH = REPORTS_METRICS_DIR / "can_st_transformer_classification_report.json"
METRICS_TEXT_PATH = REPORTS_METRICS_DIR / "can_st_transformer_classification_report.txt"
TRAIN_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "can_st_transformer_train_confusion_matrix.png"
VAL_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "can_st_transformer_val_confusion_matrix.png"
TEST_CONFUSION_MATRIX_PATH = REPORTS_FIGURES_DIR / "can_st_transformer_test_confusion_matrix.png"
TEST_CONFUSION_MATRIX_COUNTS_PATH = (
    REPORTS_FIGURES_DIR / "can_st_transformer_test_confusion_matrix_counts.png"
)
TEST_ROC_CURVE_PATH = REPORTS_FIGURES_DIR / "can_st_transformer_test_roc_curve.png"

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(
    mlflow_params.get("experiment_can_st_transformer", "can_st_transformer")
)


# ==========================================================
# Datasets
# ==========================================================

train_dataset = CANSTTransformerDataset(
    TRAIN_PATH,
    source_path=dataset_params["prepared_path"],
    window_size=window_size,
    batch_size=training_params["batch_size"],
    shuffle=False,
)
val_dataset = CANSTTransformerDataset(
    VAL_PATH,
    source_path=dataset_params["prepared_path"],
    window_size=window_size,
    batch_size=training_params["batch_size"],
    shuffle=False,
)
test_dataset = CANSTTransformerDataset(
    TEST_PATH,
    source_path=dataset_params["prepared_path"],
    window_size=window_size,
    batch_size=training_params["batch_size"],
    shuffle=False,
)
test_tf = test_dataset.to_tf_dataset()


# ==========================================================
# Load Model
# ==========================================================

model = keras.models.load_model(MODEL_PATH, compile=False)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()
print(f"\nModel loaded successfully: {MODEL_PATH}")


# ==========================================================
# Evaluate Next-Token Prediction Loss on Test
# ==========================================================

test_eval = model.evaluate(test_tf, verbose=1, return_dict=True)
test_loss = test_eval.get("loss", 0.0)


# ==========================================================
# Predict on Test Dataset (Binary Detection via Top-K)
# ==========================================================

def run_dataset_inference(dataset: CANSTTransformerDataset, sample_indices=None):
    if sample_indices is None:
        indices = np.arange(len(dataset))
    else:
        indices = np.asarray(sample_indices, dtype=np.int64)

    if len(indices) == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty((0, 6), dtype=np.int32),
            np.empty(0, dtype=np.float32),
            [],
            0.0,
        )

    all_targets = []
    all_labels = []
    all_top = []
    all_scores = []
    all_sessions = []

    t_start = perf_counter()
    max_k = 6

    for start in range(0, len(indices), dataset.batch_size):
        batch_idx = indices[start : start + dataset.batch_size]
        batch_temp = []
        batch_spat = []
        batch_tgt = []
        batch_lbl = []
        batch_sess = []

        for idx in batch_idx:
            t_in, s_in, tgt_id, lbl = dataset._window_at(int(idx))
            batch_temp.append(t_in)
            batch_spat.append(s_in)
            batch_tgt.append(tgt_id)
            batch_lbl.append(lbl)
            batch_sess.append(dataset.session_ids[idx])

        batch_temp = np.asarray(batch_temp, dtype=np.int32)
        batch_spat = np.asarray(batch_spat, dtype=np.int32)

        logits = model.predict_on_batch(
            {"temporal_input": batch_temp, "spatial_input": batch_spat}
        )

        # Softmax & Anomaly scoring
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        tgt_arr = np.asarray(batch_tgt, dtype=np.int32)
        true_probs = probs[np.arange(len(tgt_arr)), tgt_arr]
        scores = 1.0 - true_probs

        top_k = tf.math.top_k(logits, k=max_k).indices.numpy()

        all_targets.append(tgt_arr)
        all_labels.append(np.asarray(batch_lbl, dtype=np.int32))
        all_top.append(top_k)
        all_scores.append(scores)
        all_sessions.extend(batch_sess)

    t_end = perf_counter()
    elapsed = t_end - t_start

    return (
        np.concatenate(all_targets),
        np.concatenate(all_labels),
        np.concatenate(all_top, axis=0),
        np.concatenate(all_scores),
        all_sessions,
        elapsed,
    )


targets_test, y_true_test, top_test, scores_test, sessions_test, test_time = (
    run_dataset_inference(test_dataset)
)

k_thresh = range_adjustment + 1
is_normal_test = np.any(top_test[:, :k_thresh] == targets_test[:, None], axis=1)
y_pred_test = (~is_normal_test).astype(int)

ms_per_frame = (test_time / len(y_true_test) * 1000.0) if len(y_true_test) > 0 else 0.0


# ==========================================================
# Train & Validation Predictions for Confusion Matrices
# ==========================================================

train_sample_indices = stratified_sample_indices(
    (train_dataset.raw_labels > 0).astype(int), SAMPLE_SIZE, SAMPLE_SEED
)
targets_tr, y_true_tr, top_tr, _, _, _ = run_dataset_inference(
    train_dataset, train_sample_indices
)
if len(y_true_tr):
    is_normal_tr = np.any(top_tr[:, :k_thresh] == targets_tr[:, None], axis=1)
    y_pred_tr = (~is_normal_tr).astype(int)
else:
    y_pred_tr = np.empty(0, dtype=int)

if len(val_dataset):
    val_sample_indices = stratified_sample_indices(
        (val_dataset.raw_labels > 0).astype(int), SAMPLE_SIZE, SAMPLE_SEED + 1
    )
    targets_vl, y_true_vl, top_vl, _, _, _ = run_dataset_inference(
        val_dataset, val_sample_indices
    )
    is_normal_vl = np.any(top_vl[:, :k_thresh] == targets_vl[:, None], axis=1)
    y_pred_vl = (~is_normal_vl).astype(int)
else:
    y_true_vl = np.empty(0, dtype=int)
    y_pred_vl = np.empty(0, dtype=int)

train_confusion_matrix_counts, train_confusion_matrix = calculate_confusion_matrices(
    y_true_tr, y_pred_tr, len(CLASS_NAMES)
)
val_confusion_matrix_counts, val_confusion_matrix = calculate_confusion_matrices(
    y_true_vl, y_pred_vl, len(CLASS_NAMES)
)

save_confusion_matrix_figure(
    train_confusion_matrix,
    TRAIN_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "CAN ST-Transformer Train Confusion Matrix",
    percentage=True,
)
save_confusion_matrix_figure(
    val_confusion_matrix,
    VAL_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "CAN ST-Transformer Validation Confusion Matrix",
    percentage=True,
)
print(f"Train confusion matrix disimpan ke: {TRAIN_CONFUSION_MATRIX_PATH}")
print(f"Validation confusion matrix disimpan ke: {VAL_CONFUSION_MATRIX_PATH}")


# ==========================================================
# Metrics & Test Confusion Matrix
# ==========================================================

print("\n======================")
print("Evaluation Result")
print("======================")
test_metrics = calculate_classification_metrics(y_true_test, y_pred_test, CLASS_NAMES)
for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
    print(f"{key}: {test_metrics[key]:.4f}")
print("\nClassification Report\n")
print(test_metrics["classification_report_text"])

test_confusion_matrix_counts, test_confusion_matrix = calculate_confusion_matrices(
    y_true_test, y_pred_test, len(CLASS_NAMES)
)
print("\nConfusion Matrix")
print(test_confusion_matrix_counts)
save_confusion_matrix_figure(
    test_confusion_matrix_counts,
    TEST_CONFUSION_MATRIX_COUNTS_PATH,
    CLASS_NAMES,
    "CAN ST-Transformer Test Confusion Matrix Counts",
    percentage=False,
)
save_confusion_matrix_figure(
    test_confusion_matrix,
    TEST_CONFUSION_MATRIX_PATH,
    CLASS_NAMES,
    "CAN ST-Transformer Test Confusion Matrix",
    percentage=True,
)


# ==========================================================
# Additional Evaluation: Per-Attack Breakdown & Range Expansion
# ==========================================================

per_attack_evaluation = {}
unique_sessions = sorted(set(sessions_test))
sessions_arr = np.asarray(sessions_test)

for sess in unique_sessions:
    s_mask = sessions_arr == sess
    s_true = y_true_test[s_mask]
    s_pred = y_pred_test[s_mask]
    s_metrics = calculate_classification_metrics(s_true, s_pred, CLASS_NAMES)
    per_attack_evaluation[sess] = {
        "accuracy": s_metrics["accuracy"],
        "precision_macro": s_metrics["precision_macro"],
        "recall_macro": s_metrics["recall_macro"],
        "f1_macro": s_metrics["f1_macro"],
        "attack_precision": s_metrics["classification_report"].get("Attack", {}).get("precision", 0.0),
        "attack_recall": s_metrics["classification_report"].get("Attack", {}).get("recall", 0.0),
        "attack_f1": s_metrics["classification_report"].get("Attack", {}).get("f1-score", 0.0),
        "total_frames": int(np.sum(s_mask)),
    }

range_expansion_study = {}
for r in range(6):
    r_key = f"range_{r}_top_{r+1}"
    is_norm_r = np.any(top_test[:, : r + 1] == targets_test[:, None], axis=1)
    y_pred_r = (~is_norm_r).astype(int)
    m_r = calculate_classification_metrics(y_true_test, y_pred_r, CLASS_NAMES)
    range_expansion_study[r_key] = {
        "overall": {
            "precision_macro": m_r["precision_macro"],
            "recall_macro": m_r["recall_macro"],
            "f1_macro": m_r["f1_macro"],
            "attack_precision": m_r["classification_report"].get("Attack", {}).get("precision", 0.0),
            "attack_recall": m_r["classification_report"].get("Attack", {}).get("recall", 0.0),
            "attack_f1": m_r["classification_report"].get("Attack", {}).get("f1-score", 0.0),
        },
        "by_session": {},
    }
    for sess in unique_sessions:
        s_mask = sessions_arr == sess
        sm_r = calculate_classification_metrics(
            y_true_test[s_mask], y_pred_r[s_mask], CLASS_NAMES
        )
        range_expansion_study[r_key]["by_session"][sess] = {
            "attack_precision": sm_r["classification_report"].get("Attack", {}).get("precision", 0.0),
            "attack_recall": sm_r["classification_report"].get("Attack", {}).get("recall", 0.0),
            "attack_f1": sm_r["classification_report"].get("Attack", {}).get("f1-score", 0.0),
        }

# ROC Curve Plot
roc_auc_val = None
if len(np.unique(y_true_test)) > 1:
    roc_auc_val = float(roc_auc_score(y_true_test, scores_test))
    fpr, tpr, _ = roc_curve(y_true_test, scores_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Proposed ST-Transformer (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - CAN ST-Transformer (w={window_size})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(TEST_ROC_CURVE_PATH, dpi=300)
    plt.close()
    print(f"ROC curve disimpan ke: {TEST_ROC_CURVE_PATH}")


# ==========================================================
# Build & Save Standard Report
# ==========================================================

best_training_report = load_best_training_metrics(BEST_TRAINING_METRICS_PATH)
test_results_dict = {
    "loss": float(test_loss),
    "accuracy": test_metrics["accuracy"],
    "precision_macro": test_metrics["precision_macro"],
    "recall_macro": test_metrics["recall_macro"],
    "f1_macro": test_metrics["f1_macro"],
}

extra_info = {
    "inference_latency": {
        "total_eval_time_sec": float(test_time),
        "ms_per_frame": float(ms_per_frame),
    },
    "roc_auc": roc_auc_val,
    "per_attack_evaluation": per_attack_evaluation,
    "range_expansion_study": range_expansion_study,
    "window_size": window_size,
    "range_adjustment": range_adjustment,
}

full_report = build_evaluation_report(
    model_name="can_st_transformer",
    test_results=test_results_dict,
    class_names=CLASS_NAMES,
    best_training_report=best_training_report,
    test_classification_report=test_metrics["classification_report"],
    extra=extra_info,
)

save_json_report(full_report, METRICS_JSON_PATH)
save_text_report(full_report, METRICS_TEXT_PATH)

# Append extra paper tables to text report
extra_lines = [
    "",
    "=" * 70,
    "PER-ATTACK SCENARIO EVALUATION (Jo & Kim, 2024)",
    "=" * 70,
    f"{'Scenario':<32} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Frames':<10}",
    "-" * 78,
]
for sess, m in per_attack_evaluation.items():
    extra_lines.append(
        f"{sess:<32} {m['attack_precision']:<12.5f} {m['attack_recall']:<12.5f} {m['attack_f1']:<12.5f} {m['total_frames']:<10,}"
    )

extra_lines.extend([
    "",
    "=" * 70,
    "RANGE EXPANSION STUDY (Table 10 Comparison)",
    "=" * 70,
    f"{'Scenario':<28} {'Range':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}",
    "-" * 72,
])
for r in range(6):
    r_key = f"range_{r}_top_{r+1}"
    for sess in unique_sessions:
        sm = range_expansion_study[r_key]["by_session"][sess]
        extra_lines.append(
            f"{sess:<28} Range {r:<2} {sm['attack_precision']:<12.5f} {sm['attack_recall']:<12.5f} {sm['attack_f1']:<12.5f}"
        )

with open(METRICS_TEXT_PATH, "a") as f:
    f.write("\n" + "\n".join(extra_lines) + "\n")

print(f"Classification report disimpan ke: {METRICS_JSON_PATH}")
print(f"Text report disimpan ke: {METRICS_TEXT_PATH}")


# ==========================================================
# Log to MLflow (resume the training run)
# ==========================================================

artifact_paths = [
    TEST_CONFUSION_MATRIX_COUNTS_PATH,
    TEST_CONFUSION_MATRIX_PATH,
    TRAIN_CONFUSION_MATRIX_PATH,
    VAL_CONFUSION_MATRIX_PATH,
    METRICS_TEXT_PATH,
]
if TEST_ROC_CURVE_PATH.exists():
    artifact_paths.append(TEST_ROC_CURVE_PATH)

try:
    run_id = log_evaluation_to_mlflow(
        run_id=load_run_id(RUN_ID_PATH),
        report=full_report,
        report_path=METRICS_JSON_PATH,
        artifact_paths=artifact_paths,
    )
    print(f"Metrik evaluasi di-log ke MLflow run: {run_id}")
except Exception as exc:
    print(f"MLflow logging skipped: {exc}")
