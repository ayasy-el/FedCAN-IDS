"""Evaluate the Spatial-Temporal CAN Transformer IDS (Jo & Kim, 2024).

Implements the exact evaluation methodology from Jo & Kim (IEEE Access 2024):
1. Positive-Class Metrics (Equations 25, 26, 27):
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1-Score = 2 * Precision * Recall / (Precision + Recall)
2. Per-Attack Scenario Evaluation:
   - Flooding Attack (Dominant 0x000 CAN ID injection)
   - Fuzzy Attack (Random CAN ID and payload injection)
   - Malfunction Attack (Valid CAN ID with manipulated payload)
3. Range Expansion Study (Table 10):
   - Range 0 (Top-1) to Range 5 (Top-6) detection rules.
4. Latency Benchmarking (Figure 11):
   - Real-time frame processing latency (ms/frame) against the 0.26 ms SAE J2284-3 standard.
5. Multi-scenario ROC Curves & AUC Scores (Figures 12, 13, 14).
"""

import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import dagshub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras

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

# Initialize DagsHub / MLflow tracking
if os.environ.get("DAGSHUB_USER_TOKEN"):
    try:
        dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)
    except Exception as e:
        print(f"DagsHub init skipped: {e}")
else:
    try:
        dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)
    except Exception:
        pass


def compute_attack_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate exact Equations 25-27 from Jo & Kim (2024).

    - Positive class = Attack (1)
    - Negative class = Normal (0)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "total_frames": len(y_true),
        "normal_frames": int(np.sum(y_true == 0)),
        "attack_frames": int(np.sum(y_true == 1)),
    }


def main():
    params = load_params()
    dataset_params = params["dataset"]
    _, task = validate_experiment(params)
    split = params["split"]
    model_params = params["model"]["can_st_transformer"]
    training_params = params["training"]["can_st_transformer"]
    mlflow_params = params["mlflow"]

    window_size = int(model_params.get("window_size", split.get("window_size", 64)))
    range_adjustment = int(model_params.get("range_adjustment", 0))
    use_cutoff = bool(model_params.get("cutoff_frames", False))

    PAPER_CUTOFFS = {
        "Flooding_dataset_SONATA": 95_999,
        "Fuzzy_dataset_SONATA": 87_909,
        "Malfunction_dataset_SONATA": 86_999,
    }

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

    tracking_uri = mlflow_params.get("tracking_uri", "mlruns")
    if not os.environ.get("DAGSHUB_USER_TOKEN") and "dagshub.com" in str(tracking_uri):
        tracking_uri = "mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(
            mlflow_params.get("experiment_can_st_transformer", "can_st_transformer")
        )
    except Exception as e:
        print(f"MLflow experiment setup notice: {e}")

    # 1. Load Model
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print(f"\nModel loaded successfully: {MODEL_PATH}")

    # 2. Datasets
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

    # 3. Helper to run batched inference
    def run_inference(dataset: CANSTTransformerDataset, target_indices=None):
        if target_indices is None:
            indices = np.arange(len(dataset))
        else:
            indices = np.asarray(target_indices, dtype=np.int64)

        if len(indices) == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty((0, 6), dtype=np.int32),
                np.empty(0, dtype=np.float32),
                0.0,
            )

        all_targets = []
        all_labels = []
        all_top = []
        all_scores = []
        max_k = 6

        t0 = perf_counter()
        for start in range(0, len(indices), dataset.batch_size):
            batch_idx = indices[start : start + dataset.batch_size]
            batch_temp = []
            batch_spat = []
            batch_tgt = []
            batch_lbl = []

            for idx in batch_idx:
                t_in, s_in, tgt_id, lbl = dataset._window_at(int(idx))
                batch_temp.append(t_in)
                batch_spat.append(s_in)
                batch_tgt.append(tgt_id)
                batch_lbl.append(lbl)

            batch_temp = np.asarray(batch_temp, dtype=np.int32)
            batch_spat = np.asarray(batch_spat, dtype=np.int32)

            logits = model.predict_on_batch(
                {"temporal_input": batch_temp, "spatial_input": batch_spat}
            )

            # Softmax & Anomaly scores
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

        elapsed = perf_counter() - t0
        return (
            np.concatenate(all_targets),
            np.concatenate(all_labels),
            np.concatenate(all_top, axis=0),
            np.concatenate(all_scores),
            elapsed,
        )

    # 4. Run Independent Evaluation per Attack Scenario
    sessions_test = np.asarray(test_dataset.session_ids)
    unique_sessions = [
        "Flooding_dataset_SONATA",
        "Fuzzy_dataset_SONATA",
        "Malfunction_dataset_SONATA",
    ]
    # Filter only available sessions
    available_sessions = [s for s in unique_sessions if s in set(sessions_test)]
    if not available_sessions:
        available_sessions = sorted(set(sessions_test))

    # Evaluate next-token loss on test (using cutoff slice if enabled)
    test_eval_indices = []
    for s_name in available_sessions:
        s_idx = np.flatnonzero(sessions_test == s_name)
        if use_cutoff and s_name in PAPER_CUTOFFS:
            s_idx = s_idx[: max(0, PAPER_CUTOFFS[s_name] - window_size + 1)]
        test_eval_indices.extend(s_idx)
    test_eval_indices = np.asarray(test_eval_indices, dtype=np.int64)

    test_eval = model.evaluate(
        test_dataset.to_tf_dataset(test_eval_indices), verbose=1, return_dict=True
    )
    test_loss = test_eval.get("loss", 0.0)

    per_attack_results = {}
    range_expansion_study = {}
    k_thresh = range_adjustment + 1

    overall_targets = []
    overall_labels = []
    overall_top = []
    overall_scores = []
    total_eval_time = 0.0

    print("\n" + "=" * 92)
    print("RUNNING PER-ATTACK SCENARIO INFERENCE (Jo & Kim, 2024)")
    if use_cutoff:
        print(">> Cut-off frames: ENABLED (Matching Table 3 of Paper)")
    else:
        print(">> Cut-off frames: DISABLED (Full Log Evaluation)")
    print("=" * 92)

    for sess in available_sessions:
        sess_indices = np.flatnonzero(sessions_test == sess)
        if use_cutoff and sess in PAPER_CUTOFFS:
            max_frames = PAPER_CUTOFFS[sess]
            max_windows = max(0, max_frames - window_size + 1)
            sess_indices = sess_indices[:max_windows]

        print(f"\nProcessing {sess} ({len(sess_indices):,} frames)...")
        tgt, lbl, top, scr, elapsed = run_inference(test_dataset, sess_indices)
        total_eval_time += elapsed
        ms_per_frame = (elapsed / len(lbl) * 1000.0) if len(lbl) > 0 else 0.0

        overall_targets.append(tgt)
        overall_labels.append(lbl)
        overall_top.append(top)
        overall_scores.append(scr)

        # Configured Range Decision
        is_normal = np.any(top[:, :k_thresh] == tgt[:, None], axis=1)
        y_pred = (~is_normal).astype(int)

        attack_metrics = compute_attack_metrics(lbl, y_pred)
        attack_metrics["inference_seconds"] = elapsed
        attack_metrics["ms_per_frame"] = ms_per_frame
        attack_metrics["meets_realtime_spec"] = ms_per_frame <= 0.26  # SAE J2284-3 (0.26 ms)

        # ROC-AUC
        if len(np.unique(lbl)) > 1:
            try:
                attack_metrics["roc_auc"] = float(roc_auc_score(lbl, scr))
            except Exception:
                attack_metrics["roc_auc"] = None
        else:
            attack_metrics["roc_auc"] = None

        per_attack_results[sess] = attack_metrics

        # Range Expansion for this session (Range 0 to 5)
        range_expansion_study[sess] = {}
        for r in range(6):
            r_top = r + 1
            is_norm_r = np.any(top[:, :r_top] == tgt[:, None], axis=1)
            y_pred_r = (~is_norm_r).astype(int)
            r_m = compute_attack_metrics(lbl, y_pred_r)
            range_expansion_study[sess][f"range_{r}"] = {
                "range": r,
                "top_k": r_top,
                "precision": r_m["precision"],
                "recall": r_m["recall"],
                "f1": r_m["f1"],
                "accuracy": r_m["accuracy"],
                "fp": r_m["fp"],
                "fn": r_m["fn"],
            }

    # 5. Overall Test Aggregation
    targets_all = np.concatenate(overall_targets)
    labels_all = np.concatenate(overall_labels)
    top_all = np.concatenate(overall_top, axis=0)
    scores_all = np.concatenate(overall_scores)

    is_normal_all = np.any(top_all[:, :k_thresh] == targets_all[:, None], axis=1)
    y_pred_all = (~is_normal_all).astype(int)
    overall_attack_metrics = compute_attack_metrics(labels_all, y_pred_all)
    overall_ms_per_frame = (total_eval_time / len(labels_all) * 1000.0) if len(labels_all) > 0 else 0.0
    overall_attack_metrics["inference_seconds"] = total_eval_time
    overall_attack_metrics["ms_per_frame"] = overall_ms_per_frame
    overall_attack_metrics["meets_realtime_spec"] = overall_ms_per_frame <= 0.26

    if len(np.unique(labels_all)) > 1:
        overall_attack_metrics["roc_auc"] = float(roc_auc_score(labels_all, scores_all))
    else:
        overall_attack_metrics["roc_auc"] = None

    # Overall Range Expansion
    overall_range_study = {}
    for r in range(6):
        r_top = r + 1
        is_norm_r = np.any(top_all[:, :r_top] == targets_all[:, None], axis=1)
        y_pred_r = (~is_norm_r).astype(int)
        r_m = compute_attack_metrics(labels_all, y_pred_r)
        overall_range_study[f"range_{r}"] = {
            "range": r,
            "top_k": r_top,
            "precision": r_m["precision"],
            "recall": r_m["recall"],
            "f1": r_m["f1"],
            "accuracy": r_m["accuracy"],
            "fp": r_m["fp"],
            "fn": r_m["fn"],
        }
    range_expansion_study["OVERALL"] = overall_range_study

    # 6. Train & Val Sample Predictions for Compatibility Matrices
    train_sample_indices = stratified_sample_indices(
        (train_dataset.raw_labels > 0).astype(int), SAMPLE_SIZE, SAMPLE_SEED
    )
    targets_tr, y_true_tr, top_tr, _, _ = run_inference(train_dataset, train_sample_indices)
    if len(y_true_tr):
        is_normal_tr = np.any(top_tr[:, :k_thresh] == targets_tr[:, None], axis=1)
        y_pred_tr = (~is_normal_tr).astype(int)
    else:
        y_pred_tr = np.empty(0, dtype=int)

    train_cm_counts, train_cm_pct = (
        calculate_confusion_matrices(y_true_tr, y_pred_tr, len(CLASS_NAMES))
        if len(y_true_tr) > 0
        else (np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=float))
    )
    val_cm_counts = np.zeros((2, 2), dtype=int)
    val_cm_pct = np.zeros((2, 2), dtype=float)

    save_confusion_matrix_figure(
        train_cm_pct,
        TRAIN_CONFUSION_MATRIX_PATH,
        CLASS_NAMES,
        "CAN ST-Transformer Train Confusion Matrix",
        percentage=True,
    )
    save_confusion_matrix_figure(
        val_cm_pct,
        VAL_CONFUSION_MATRIX_PATH,
        CLASS_NAMES,
        "CAN ST-Transformer Validation Confusion Matrix",
        percentage=True,
    )

    # Save Test Confusion Matrices
    test_cm_counts, test_cm_pct = calculate_confusion_matrices(
        labels_all, y_pred_all, len(CLASS_NAMES)
    )
    save_confusion_matrix_figure(
        test_cm_counts,
        TEST_CONFUSION_MATRIX_COUNTS_PATH,
        CLASS_NAMES,
        "CAN ST-Transformer Test Confusion Matrix Counts",
        percentage=False,
    )
    save_confusion_matrix_figure(
        test_cm_pct,
        TEST_CONFUSION_MATRIX_PATH,
        CLASS_NAMES,
        "CAN ST-Transformer Test Confusion Matrix",
        percentage=True,
    )

    # 7. Multi-Scenario ROC Curves Plot (Figures 12, 13, 14 in Paper)
    colors = {
        "Flooding_dataset_SONATA": "#d62728",      # Red
        "Fuzzy_dataset_SONATA": "#1f77b4",         # Blue
        "Malfunction_dataset_SONATA": "#2ca02c",   # Green
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for sess, tgt, lbl, scr in zip(
        available_sessions, overall_targets, overall_labels, overall_scores
    ):
        if len(np.unique(lbl)) > 1:
            fpr, tpr, _ = roc_curve(lbl, scr)
            auc = per_attack_results[sess]["roc_auc"]
            short_name = sess.replace("_dataset_SONATA", "").capitalize()
            ax.plot(
                fpr,
                tpr,
                label=f"{short_name} (AUC = {auc:.4f})" if auc else short_name,
                color=colors.get(sess, None),
                linewidth=2.0,
            )

    if len(np.unique(labels_all)) > 1:
        fpr_all, tpr_all, _ = roc_curve(labels_all, scores_all)
        ax.plot(
            fpr_all,
            tpr_all,
            label=f"Overall (AUC = {overall_attack_metrics['roc_auc']:.4f})",
            color="black",
            linestyle="--",
            linewidth=1.8,
        )

    ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Guess")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax.set_title(
        f"ROC Curves - Spatial-Temporal CAN Transformer (w={window_size})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(TEST_ROC_CURVE_PATH, dpi=300)
    plt.close(fig)
    print(f"\nMulti-scenario ROC curve saved to: {TEST_ROC_CURVE_PATH}")

    # 8. Print Prominent Paper-Style Summary
    print("\n" + "=" * 96)
    print(f"PER-ATTACK EVALUATION SUMMARY (Jo & Kim 2024 - Table 6/9 Benchmark)")
    print(f"Window Size: {window_size} | Prediction Range: Range {range_adjustment} (Top-{k_thresh})")
    print("=" * 96)
    print(
        f"{'Attack Scenario':<32} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<10} {'Latency (ms)':<14}"
    )
    print("-" * 96)
    for sess in available_sessions:
        m = per_attack_results[sess]
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        print(
            f"{sess:<32} {m['precision']:<12.5f} {m['recall']:<12.5f} {m['f1']:<12.5f} {auc_str:<10} {m['ms_per_frame']:<14.4f}"
        )
    print("-" * 96)
    ov_auc_str = (
        f"{overall_attack_metrics['roc_auc']:.4f}"
        if overall_attack_metrics["roc_auc"] is not None
        else "N/A"
    )
    print(
        f"{'OVERALL COMBINED':<32} {overall_attack_metrics['precision']:<12.5f} {overall_attack_metrics['recall']:<12.5f} {overall_attack_metrics['f1']:<12.5f} {ov_auc_str:<10} {overall_attack_metrics['ms_per_frame']:<14.4f}"
    )
    print("=" * 96)

    # Print Range Expansion Summary (Table 10)
    print("\n" + "=" * 96)
    print("RANGE EXPANSION STUDY (Table 10 Comparison: Top-1 to Top-6)")
    print("=" * 96)
    print(f"{'Attack Scenario':<28} {'Range':<10} {'Precision':<14} {'Recall':<14} {'F1-Score':<14}")
    print("-" * 80)
    for r in range(6):
        for sess in available_sessions:
            sm = range_expansion_study[sess][f"range_{r}"]
            print(
                f"{sess:<28} Range {r} (Top-{r+1})  {sm['precision']:<14.5f} {sm['recall']:<14.5f} {sm['f1']:<14.5f}"
            )
        print("-" * 80)

    # 9. Build Standard JSON and Text Reports
    std_metrics = calculate_classification_metrics(labels_all, y_pred_all, CLASS_NAMES)
    best_training_report = load_best_training_metrics(BEST_TRAINING_METRICS_PATH)
    test_results_dict = {
        "loss": float(test_loss),
        "accuracy": overall_attack_metrics["accuracy"],
        "precision_macro": std_metrics["precision_macro"],
        "recall_macro": std_metrics["recall_macro"],
        "f1_macro": std_metrics["f1_macro"],
    }
    extra_info = {
        "paper_methodology": "Jo & Kim (IEEE Access 2024)",
        "cutoff_frames_enabled": use_cutoff,
        "window_size": window_size,
        "range_adjustment": range_adjustment,
        "overall_attack_metrics": overall_attack_metrics,
        "per_attack_results": per_attack_results,
        "range_expansion_study": range_expansion_study,
    }

    full_report = build_evaluation_report(
        model_name="can_st_transformer",
        test_results=test_results_dict,
        class_names=CLASS_NAMES,
        best_training_report=best_training_report,
        test_classification_report=std_metrics["classification_report"],
        extra=extra_info,
    )
    save_json_report(full_report, METRICS_JSON_PATH)
    save_text_report(full_report, METRICS_TEXT_PATH)

    # Append per-attack tables to text report
    lines = [
        "",
        "=" * 88,
        "PER-ATTACK SCENARIO EVALUATION (Jo & Kim, 2024 - Table 6/9)",
        f"Window Size: {window_size} | Range: {range_adjustment} (Top-{k_thresh}) | Cutoff: {'ENABLED (Table 3 limits)' if use_cutoff else 'DISABLED'}",
        "=" * 88,
        f"{'Scenario':<32} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<10} {'Latency (ms)':<12}",
        "-" * 88,
    ]
    for sess in available_sessions:
        m = per_attack_results[sess]
        auc_s = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "N/A"
        lines.append(
            f"{sess:<32} {m['precision']:<12.5f} {m['recall']:<12.5f} {m['f1']:<12.5f} {auc_s:<10} {m['ms_per_frame']:<12.4f}"
        )
    lines.append("-" * 88)
    lines.append(
        f"{'OVERALL':<32} {overall_attack_metrics['precision']:<12.5f} {overall_attack_metrics['recall']:<12.5f} {overall_attack_metrics['f1']:<12.5f} {ov_auc_str:<10} {overall_attack_metrics['ms_per_frame']:<12.4f}"
    )

    lines.extend([
        "",
        "=" * 88,
        "RANGE EXPANSION STUDY (Jo & Kim, 2024 - Table 10)",
        "=" * 88,
        f"{'Scenario':<28} {'Range':<10} {'Precision':<14} {'Recall':<14} {'F1-Score':<14}",
        "-" * 80,
    ])
    for r in range(6):
        for sess in available_sessions:
            sm = range_expansion_study[sess][f"range_{r}"]
            lines.append(
                f"{sess:<28} Range {r} (Top-{r+1})  {sm['precision']:<14.5f} {sm['recall']:<14.5f} {sm['f1']:<14.5f}"
            )
        print("-" * 80)

    with open(METRICS_TEXT_PATH, "a") as f:
        f.write("\n" + "\n".join(lines) + "\n")

    print(f"Classification report saved to: {METRICS_JSON_PATH}")
    print(f"Text report saved to: {METRICS_TEXT_PATH}")

    # 10. MLflow Logging
    mlflow_metrics = {
        "test_loss": float(test_loss),
        "overall_attack_precision": overall_attack_metrics["precision"],
        "overall_attack_recall": overall_attack_metrics["recall"],
        "overall_attack_f1": overall_attack_metrics["f1"],
        "overall_latency_ms_per_frame": overall_ms_per_frame,
    }
    if overall_attack_metrics["roc_auc"] is not None:
        mlflow_metrics["overall_roc_auc"] = overall_attack_metrics["roc_auc"]

    for sess in available_sessions:
        short_name = sess.replace("_dataset_SONATA", "").lower()
        m = per_attack_results[sess]
        mlflow_metrics[f"{short_name}_precision"] = m["precision"]
        mlflow_metrics[f"{short_name}_recall"] = m["recall"]
        mlflow_metrics[f"{short_name}_f1"] = m["f1"]
        mlflow_metrics[f"{short_name}_latency_ms"] = m["ms_per_frame"]
        if m["roc_auc"] is not None:
            mlflow_metrics[f"{short_name}_roc_auc"] = m["roc_auc"]

    artifact_paths = [
        TEST_CONFUSION_MATRIX_COUNTS_PATH,
        TEST_CONFUSION_MATRIX_PATH,
        TRAIN_CONFUSION_MATRIX_PATH,
        VAL_CONFUSION_MATRIX_PATH,
        TEST_ROC_CURVE_PATH,
        METRICS_TEXT_PATH,
    ]

    try:
        run_id = log_evaluation_to_mlflow(
            run_id=load_run_id(RUN_ID_PATH),
            report=full_report,
            report_path=METRICS_JSON_PATH,
            artifact_paths=artifact_paths,
        )
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(mlflow_metrics)
            print(f"Evaluation metrics logged to MLflow run: {run_id}")
    except Exception as exc:
        print(f"MLflow logging notice: {exc}")


if __name__ == "__main__":
    main()
