"""Imperative utility to sync all pre-training, round, and evaluation artifacts to DagsHub MLflow."""

import argparse
import ast
import json
import re
from pathlib import Path

import mlflow

from utils.mlflow_utils import init_dagshub, load_run_id, log_artifact_organized, write_model_report
from utils.params import load_params


def parse_metrics_from_log(log_path: Path) -> dict:
    """Extract round history and test metrics from simulation output log if available."""
    if not log_path.exists():
        return {}

    content = log_path.read_text(encoding="utf-8", errors="replace")
    parsed: dict = {"rounds": {}, "test_metrics": {}, "run_id": None}

    # Extract Run ID
    run_match = re.search(r"runs/([a-f0-9]{32})", content)
    if run_match:
        parsed["run_id"] = run_match.group(1)

    # Extract History dict at end of Flower simulation
    history_match = re.search(r"History \(metrics, centralized\):\s*(\{.*?\})\s*INFO", content, re.DOTALL)
    if history_match:
        try:
            # Clean up ray/info tags if any
            clean_dict_str = re.sub(r"\[\d+m.*?\[0m", "", history_match.group(1))
            clean_dict_str = re.sub(r"INFO\s*:\s*", "", clean_dict_str)
            metrics_dict = ast.literal_eval(clean_dict_str)
            for metric_name, values in metrics_dict.items():
                for round_num, val in values:
                    round_num = int(round_num)
                    parsed["rounds"].setdefault(round_num, {})[metric_name] = float(val)
        except Exception as e:
            print(f"Notice: Could not parse history dictionary: {e}")

    # Fallback/complement with per-round regex
    for m in re.finditer(r"\[Round\s+(\d+)\]\s+Server Centralized Eval:\s+Loss=([\d.]+),\s+Acc=([\d.]+),\s+F1-Macro=([\d.]+)", content):
        rnd = int(m.group(1))
        parsed["rounds"].setdefault(rnd, {})["val_loss"] = float(m.group(2))
        parsed["rounds"].setdefault(rnd, {})["val_accuracy"] = float(m.group(3))
        parsed["rounds"].setdefault(rnd, {})["val_f1_macro"] = float(m.group(4))

    # Parse per-client round metrics
    # e.g.: Participating: Client 0 (loss=0.0272, acc=0.9913, f1=0.9899), ...
    for m in re.finditer(r"\[Round Summary - Clients Fit\]\s+Avg Loss=([\d.]+)\s+\|\s+Avg Acc=([\d.]+)\s*\n\s*Participating:\s*([^\n]+)", content):
        avg_loss = float(m.group(1))
        avg_acc = float(m.group(2))
        details = m.group(3)
        client_matches = re.finditer(r"Client\s+(\d+)\s+\(loss=([\d.]+),\s+acc=([\d.]+)(?:,\s+f1=([\d.]+))?\)", details)
        client_dict = {
            "train_avg_loss": avg_loss,
            "train_avg_accuracy": avg_acc,
        }
        for cm in client_matches:
            cid = cm.group(1)
            client_dict[f"client_{cid}_loss"] = float(cm.group(2))
            client_dict[f"client_{cid}_accuracy"] = float(cm.group(3))
            if cm.group(4):
                client_dict[f"client_{cid}_f1_macro"] = float(cm.group(4))
        # Attach to the latest round found so far
        if parsed["rounds"]:
            latest_rnd = max(parsed["rounds"].keys())
            parsed["rounds"][latest_rnd].update(client_dict)

    # Extract test metrics
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        tm = re.search(rf"{metric.replace('_', ' ').capitalize()}:\s*([\d.]+)", content, re.IGNORECASE)
        if tm:
            parsed["test_metrics"][f"test_{metric}"] = float(tm.group(1))

    lat_m = re.search(r"Single-sample inference latency:\s*([\d.]+)\s*ms", content)
    if lat_m:
        parsed["test_metrics"]["inference_latency_ms"] = float(lat_m.group(1))

    return parsed


def sync_fl_run(model_name: str = "hierarchical_transformer_gru", log_file: str = "log.txt") -> None:
    """Sync all federated artifacts, metrics, and models to DagsHub MLflow."""
    params = load_params()
    init_dagshub()

    dataset_params = params["dataset"]
    processed_dir = Path(dataset_params["processed_dir"])
    mlflow_params = params.get("mlflow", {})
    tracking_uri = mlflow_params.get("tracking_uri", "https://dagshub.com/ayasy-el/FedCAN-IDS.mlflow")
    experiment_name = mlflow_params.get("experiment_fl", "federated_can_ids")

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"MLflow experiment notice: {e}")

    # Determine run ID
    run_id_file = Path("checkpoints") / f"fl_{model_name}_mlflow_run_id.txt"
    run_id = load_run_id(run_id_file)

    log_path = Path(log_file)
    parsed = parse_metrics_from_log(log_path) if log_path.exists() else {}

    if not run_id and parsed.get("run_id"):
        run_id = parsed["run_id"]

    if run_id:
        print(f"[Sync] Found existing MLflow Run ID: {run_id}")
    else:
        print("[Sync] No existing run ID found; creating new MLflow run for artifacts.")

    # 1. Log parsed per-round and test metrics
    with mlflow.start_run(run_id=run_id, run_name=f"fl_{model_name}") as run:
        active_id = run.info.run_id

        # Update run_id_file if not already set
        if not run_id_file.exists() or not run_id_file.read_text().strip():
            run_id_file.parent.mkdir(parents=True, exist_ok=True)
            run_id_file.write_text(active_id)

        # Log round metrics
        if parsed.get("rounds"):
            print(f"[Sync] Logging per-round curves for {len(parsed['rounds'])} rounds...")
            for rnd, m_dict in sorted(parsed["rounds"].items()):
                mlflow.log_metrics(m_dict, step=rnd)

        # Log test metrics
        if parsed.get("test_metrics"):
            print(f"[Sync] Logging test evaluation metrics: {parsed['test_metrics']}")
            mlflow.log_metrics(parsed["test_metrics"])

        # 2. Ensure Model Report artifact exists
        model_report_path = Path("reports/model_summaries") / f"fl_{model_name}_model_report.txt"
        if not model_report_path.exists():
            model_report_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from fl.models import get_model_and_dataset
                val_path = processed_dir / "val.parquet"
                model, _ = get_model_and_dataset(model_name, val_path, params, is_training=False)
                write_model_report(
                    path=model_report_path,
                    model_name=f"fl_{model_name}",
                    model=model,
                    sections={
                        "dataset": dataset_params,
                        "split": params.get("split", {}),
                        "federated": params.get("federated", {}),
                        "model": params.get("model", {}).get(model_name, {}),
                        "training": params.get("training", {}).get(model_name, {}),
                    },
                )
            except Exception as e:
                print(f"Notice: Could not auto-generate model report: {e}")

        # 3. Collect and upload all artifacts
        artifacts_to_upload = [
            # Pre-training
            (model_report_path, "reports/model_summaries"),
            (processed_dir / "federated" / "partition_summary.json", "federated"),
            # Models
            (Path("checkpoints") / f"fl_{model_name}_best.keras", "models"),
            (Path("checkpoints") / f"fl_{model_name}_final.keras", "models"),
            (run_id_file, "checkpoints"),
            # Metrics
            (Path("reports/metrics") / f"fl_{model_name}_metrics.json", "metrics"),
            (Path("reports/metrics") / f"fl_{model_name}_best_metrics.json", "metrics"),
            (Path("reports/metrics") / f"fl_{model_name}_clients_history.json", "metrics"),
            # Reports
            (Path("reports/metrics") / f"fl_{model_name}_classification_report.json", "reports"),
            (Path("reports/metrics") / f"fl_{model_name}_classification_report.txt", "reports"),
            # Figures
            (Path("reports/figures") / f"fl_{model_name}_test_confusion_matrix.png", "figures/confusion_matrices"),
            (Path("reports/figures") / f"fl_{model_name}_test_confusion_matrix_counts.png", "figures/confusion_matrices"),
            (Path("reports/figures") / f"fl_{model_name}_train_confusion_matrix.png", "figures/confusion_matrices"),
            (Path("reports/figures") / f"fl_{model_name}_val_confusion_matrix.png", "figures/confusion_matrices"),
        ]

        uploaded_count = 0
        for path_obj, target_dir in artifacts_to_upload:
            if path_obj.exists():
                log_artifact_organized(path_obj, target_dir)
                uploaded_count += 1
                print(f"  [Artifact] Uploaded: {path_obj.name} -> {target_dir}")
            else:
                print(f"  [Notice] File not on disk (skipped): {path_obj}")

        print(f"\n[Sync] Successfully uploaded {uploaded_count} artifacts to DagsHub MLflow run: {active_id}")
        run_url = f"{tracking_uri.removesuffix('.mlflow')}/src/branch/master"
        print(f"[Sync] View run at: {tracking_uri}/#/experiments/18/runs/{active_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync FL artifacts to DagsHub")
    parser.add_argument("--model", type=str, default="hierarchical_transformer_gru", help="Model name")
    parser.add_argument("--log", type=str, default="log.txt", help="Log file path")
    args = parser.parse_args()

    sync_fl_run(model_name=args.model, log_file=args.log)
