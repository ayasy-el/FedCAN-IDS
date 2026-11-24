"""Evaluation script for Federated Learning models on test data."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from tensorflow import keras

from data.task import validate_experiment
from fl.models import get_model_and_dataset
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
from utils.mlflow_utils import init_dagshub, load_run_id
from utils.params import load_params


def evaluate_federated_model(model_name: str = "hierarchical_transformer_gru", params: dict | None = None) -> None:
    if params is None:
        params = load_params()

    init_dagshub()

    dataset_params = params["dataset"]
    task = validate_experiment(params, model_name)
    num_classes = task["num_classes"]
    class_names = task["class_names"]
    split = params.get("split", {})
    model_params = params.get("model", {}).get(model_name, {})
    fl_params = params.get("federated", {})
    mlflow_params = params.get("mlflow", {})

    processed_dir = Path(dataset_params["processed_dir"])
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    model_path = Path("checkpoints") / f"fl_{model_name}_best.keras"
    run_id_path = Path("checkpoints") / f"fl_{model_name}_mlflow_run_id.txt"
    best_metrics_path = Path("reports/metrics") / f"fl_{model_name}_best_metrics.json"

    reports_metrics_dir = Path("reports/metrics")
    reports_figures_dir = Path("reports/figures")
    reports_metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = reports_metrics_dir / f"fl_{model_name}_classification_report.json"
    metrics_text_path = reports_metrics_dir / f"fl_{model_name}_classification_report.txt"
    train_cm_path = reports_figures_dir / f"fl_{model_name}_train_confusion_matrix.png"
    val_cm_path = reports_figures_dir / f"fl_{model_name}_val_confusion_matrix.png"
    test_cm_path = reports_figures_dir / f"fl_{model_name}_test_confusion_matrix.png"
    test_cm_counts_path = reports_figures_dir / f"fl_{model_name}_test_confusion_matrix_counts.png"

    tracking_uri = mlflow_params.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_params.get("experiment_fl", "federated_can_ids")
    mlflow.set_experiment(experiment_name)

    # 1. Load Datasets
    print(f"\n[Eval] Loading datasets for {model_name}...")
    _, test_dataset = get_model_and_dataset(
        model_name=model_name,
        data_path=test_path,
        params=params,
        is_training=False,
    )
    _, val_dataset = get_model_and_dataset(
        model_name=model_name,
        data_path=val_path,
        params=params,
        is_training=False,
    )
    _, train_dataset = get_model_and_dataset(
        model_name=model_name,
        data_path=train_path,
        params=params,
        is_training=False,
    )

    # 2. Load Model
    if not model_path.exists():
        raise FileNotFoundError(f"Federated best model checkpoint not found: {model_path}")

    # Build blank architecture and load weights
    model, _ = get_model_and_dataset(
        model_name=model_name,
        data_path=test_path,
        params=params,
        is_training=False,
    )
    try:
        model.load_weights(str(model_path))
    except Exception:
        # Fallback to keras.models.load_model
        model = keras.models.load_model(str(model_path), compile=False)

    print(f"[Eval] Model loaded successfully from {model_path}")

    # 3. Latency Benchmark
    latency_samples = min(100, len(test_dataset.y))
    latency_durations = []
    if latency_samples > 0:
        for idx in range(latency_samples):
            w = test_dataset._window_at(idx)
            # Handle tuple output differences across datasets
            if isinstance(w, tuple) and len(w) == 2:
                sample_input = [np.expand_dims(x, 0) for x in w]
            elif isinstance(w, tuple) and len(w) == 4:
                sample_input = [np.expand_dims(w[0], 0), np.expand_dims(w[1], 0)]
            else:
                sample_input = np.expand_dims(w, 0)

            t0 = perf_counter()
            _ = model(sample_input, training=False)
            latency_durations.append((perf_counter() - t0) * 1000.0)

        mean_latency_ms = float(np.mean(latency_durations[5:])) if len(latency_durations) > 5 else float(np.mean(latency_durations))
        print(f"[Eval] Single-sample inference latency: {mean_latency_ms:.4f} ms")
    else:
        mean_latency_ms = None

    # 4. Evaluate on Test Set
    test_tf = test_dataset.to_tf_dataset()
    test_results = model.evaluate(test_tf, verbose=1, return_dict=True)

    print("\n[Eval] Global Model Test Results:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")

    predictions = model.predict(test_tf, verbose=1)
    y_true = test_dataset.y
    y_pred = np.argmax(np.asarray(predictions), axis=-1)

    # 5. Confusion Matrices
    test_cm_counts, test_cm = calculate_confusion_matrices(y_true, y_pred, len(class_names))
    save_confusion_matrix_figure(
        test_cm,
        test_cm_path,
        class_names,
        f"FL {model_name} Test Confusion Matrix",
        percentage=True,
    )
    save_confusion_matrix_figure(
        test_cm_counts,
        test_cm_counts_path,
        class_names,
        f"FL {model_name} Test Confusion Matrix (Counts)",
        percentage=False,
    )

    # Train / Val sample confusion matrices
    sample_size = 5000
    train_sample_idx = stratified_sample_indices(train_dataset.y, sample_size, 42)
    train_preds = np.argmax(model.predict(train_dataset.to_tf_dataset(train_sample_idx), verbose=0), axis=-1)
    _, train_cm = calculate_confusion_matrices(train_dataset.y[train_sample_idx], train_preds, len(class_names))
    save_confusion_matrix_figure(
        train_cm, train_cm_path, class_names, f"FL {model_name} Train Confusion Matrix", percentage=True
    )

    if len(val_dataset.y) > 0:
        val_sample_idx = stratified_sample_indices(val_dataset.y, sample_size, 43)
        val_preds = np.argmax(model.predict(val_dataset.to_tf_dataset(val_sample_idx), verbose=0), axis=-1)
        _, val_cm = calculate_confusion_matrices(val_dataset.y[val_sample_idx], val_preds, len(class_names))
        save_confusion_matrix_figure(
            val_cm, val_cm_path, class_names, f"FL {model_name} Val Confusion Matrix", percentage=True
        )

    # 6. Classification Metrics Report
    test_metrics = calculate_classification_metrics(y_true, y_pred, class_names)
    test_results.setdefault("precision_macro", test_metrics["precision_macro"])
    test_results.setdefault("recall_macro", test_metrics["recall_macro"])
    test_results.setdefault("f1_macro", test_metrics["f1_macro"])

    print(f"\nAccuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision Macro: {test_metrics['precision_macro']:.4f}")
    print(f"Recall Macro: {test_metrics['recall_macro']:.4f}")
    print(f"F1 Macro: {test_metrics['f1_macro']:.4f}\n")
    print(test_metrics["classification_report_text"])

    best_tr_metrics = load_best_training_metrics(best_metrics_path) if best_metrics_path.exists() else None
    param_counts = {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(int(w.numpy().size) for w in model.trainable_weights)),
    }

    full_report = build_evaluation_report(
        model_name=f"fl_{model_name}",
        test_results=test_results,
        class_names=class_names,
        best_training_report=best_tr_metrics,
        test_classification_report=test_metrics["classification_report"],
        extra={
            "dataset": dataset_params,
            "split": split,
            "model": model_params,
            "federated": fl_params,
            "parameter_counts": param_counts,
            "mean_latency_ms": mean_latency_ms,
        },
    )

    save_json_report(full_report, metrics_json_path)
    save_text_report(full_report, metrics_text_path)

    # 7. Log to MLflow
    run_id = load_run_id(run_id_path) if run_id_path.exists() else None
    if run_id:
        log_evaluation_to_mlflow(
            run_id=run_id,
            test_results=test_results,
            test_metrics=test_metrics,
            full_report=full_report,
            json_report_path=metrics_json_path,
            text_report_path=metrics_text_path,
            figures=[
                (train_cm_path, "figures/confusion_matrices"),
                (val_cm_path, "figures/confusion_matrices"),
                (test_cm_path, "figures/confusion_matrices"),
                (test_cm_counts_path, "figures/confusion_matrices"),
            ],
        )
        print(f"[Eval] Attached evaluation results to MLflow run {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Federated Learning Model")
    parser.add_argument(
        "--model",
        type=str,
        default="hierarchical_transformer_gru",
        help="Model architecture: 'hierarchical_transformer_gru', 'cnn_lstm', 'transformer_gru'",
    )
    args = parser.parse_args()

    evaluate_federated_model(model_name=args.model)
