"""Shared evaluation, confusion-matrix, and report utilities."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.mlflow_utils import log_artifact_organized


def stratified_sample_indices(y, max_samples: int, seed: int) -> np.ndarray:
    """Select at most ``max_samples`` rows while preserving class proportions."""
    if len(y) <= max_samples:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_counts = counts * max_samples / len(y)
    sample_counts = np.floor(target_counts).astype(int)
    remainder = max_samples - int(sample_counts.sum())
    order = np.argsort(-(target_counts - sample_counts))
    sample_counts[order[:remainder]] += 1

    selected = []
    for cls, count in zip(classes, sample_counts):
        class_indices = np.flatnonzero(y == cls)
        selected.append(rng.choice(class_indices, size=count, replace=False))

    selected = np.concatenate(selected)
    rng.shuffle(selected)
    return selected


def calculate_confusion_matrices(y_true, y_pred, num_classes: int):
    """Return raw-count and row-normalized percentage confusion matrices."""
    counts = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    percentage = (
        counts.astype(float) / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 100
    )
    return counts, percentage


def save_confusion_matrix_figure(
    matrix,
    path: Path,
    class_names: list[str],
    title: str,
    percentage: bool,
):
    """Save one confusion matrix figure using the repository's common style."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if percentage else "d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        vmin=0 if percentage else None,
        vmax=100 if percentage else None,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def calculate_classification_metrics(y_true, y_pred, class_names: list[str]):
    """Calculate scalar and per-class classification metrics."""
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "classification_report_text": report_text,
    }


def load_best_training_metrics(path: str | Path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def build_metric_comparison(best_training_report, test_metrics):
    if best_training_report is None:
        return None

    best_metrics = best_training_report.get("best_metrics", {})
    metric_names = ("loss", "accuracy", "precision_macro", "recall_macro", "f1_macro")
    return {
        "best_epoch": best_training_report.get("best_epoch"),
        "best_train": {key: best_metrics.get(key) for key in metric_names},
        "best_validation": {
            key: best_metrics.get(f"val_{key}") for key in metric_names
        },
        "test": test_metrics,
    }


def build_evaluation_report(
    *,
    model_name: str,
    test_results: dict,
    class_names: list[str],
    best_training_report=None,
    test_classification_report=None,
    extra: dict | None = None,
):
    """Build the common JSON report structure used by both evaluators."""
    test_metrics = {
        "loss": float(test_results["loss"]),
        "accuracy": float(test_results["accuracy"]),
        "precision_macro": float(test_results["precision_macro"]),
        "recall_macro": float(test_results["recall_macro"]),
        "f1_macro": float(test_results["f1_macro"]),
    }
    report = {
        "model": model_name,
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "f1_macro": test_metrics["f1_macro"],
            "classification_report": test_classification_report,
        },
        "class_names": class_names,
    }
    if best_training_report is not None:
        report["comparison"] = build_metric_comparison(
            best_training_report, test_metrics
        )
    if extra:
        report.update(extra)
    return report


def save_json_report(report: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def save_text_report(report: dict, path: str | Path):
    """Save a compact, human-readable version of the evaluation report."""
    test = report["test"]
    lines = [
        f"Model: {report['model']}",
        "",
        "TEST METRICS",
        "------------",
    ]
    for key in ("loss", "accuracy", "precision_macro", "recall_macro", "f1_macro"):
        lines.append(f"{key:20}: {test[key]:.6f}")

    comparison = report.get("comparison")
    if comparison:
        lines.extend(["", "CHECKPOINT VS TEST", "-------------------"])
        rows = [
            ("best_train", comparison.get("best_train", {})),
            ("best_validation", comparison.get("best_validation", {})),
            ("test", comparison.get("test", {})),
        ]
        lines.append(f"Best checkpoint epoch: {comparison.get('best_epoch')}")
        lines.append("")
        lines.append(
            "split                      loss       accuracy   precision   recall      f1"
        )
        for name, values in rows:
            lines.append(
                f"{name:28} "
                f"{values.get('loss', 0):.6f}   "
                f"{values.get('accuracy', 0):.6f}   "
                f"{values.get('precision_macro', 0):.6f}   "
                f"{values.get('recall_macro', 0):.6f}   "
                f"{values.get('f1_macro', 0):.6f}"
            )

    classification_report = test.get("classification_report")
    if classification_report:
        lines.extend(["", "CLASSIFICATION REPORT", "----------------------"])
        lines.append(
            "class                 precision   recall      f1          support"
        )
        for class_name, values in classification_report.items():
            if not isinstance(values, dict) or "precision" not in values:
                continue
            lines.append(
                f"{class_name:22} "
                f"{values['precision']:.6f}   "
                f"{values['recall']:.6f}   "
                f"{values['f1-score']:.6f}   "
                f"{int(values['support']):>8,}"
            )

    for split in ("train", "eval", "test"):
        matrix_key = f"{split}_confusion_matrix"
        counts_key = f"{split}_confusion_matrix_counts"
        if matrix_key not in report:
            continue
        lines.extend(["", f"{split.upper()} CONFUSION MATRIX (%)", "-" * 28])
        lines.append("true \\ predicted")
        lines.extend(_format_matrix(report["class_names"], report[matrix_key]))
        if counts_key in report:
            lines.extend(["", f"{split.upper()} CONFUSION MATRIX (COUNTS)", "-" * 35])
            lines.append("true \\ predicted")
            lines.extend(_format_matrix(report["class_names"], report[counts_key]))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _format_matrix(class_names, matrix):
    header = "true \\ predicted | " + " | ".join(f"{name:>10}" for name in class_names)
    separator = "-" * len(header)
    rows = [header, separator]
    for name, values in zip(class_names, matrix):
        rows.append(
            f"{name:17} | "
            + " | ".join(
                f"{float(value):10.2f}"
                if isinstance(value, float)
                else f"{int(value):10d}"
                for value in values
            )
        )
    return rows


def log_evaluation_to_mlflow(
    *,
    run_id: str | None,
    report: dict,
    report_path: str | Path,
    artifact_paths: list[str | Path],
):
    """Log standardized test metrics and report artifacts to the active run."""
    with mlflow.start_run(run_id=run_id) as run:
        if run_id is None:
            print("WARNING: MLflow run ID tidak ditemukan -- memakai run baru.")

        log_artifact_organized(report_path, "reports")
        for artifact_path in artifact_paths:
            log_artifact_organized(artifact_path)

    return run.info.run_id
