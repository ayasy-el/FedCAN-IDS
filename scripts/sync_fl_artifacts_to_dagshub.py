"""Imperative utility to sync all local epoch metrics, round metrics, and artifacts from completed FL runs to DagsHub MLflow."""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from utils.mlflow_utils import init_dagshub, load_run_id, log_artifact_organized, write_model_report
from utils.params import load_params


def parse_metrics_from_log(log_path: Path) -> dict:
    """Extract all local epochs, round history, and test evaluation metrics from log.txt."""
    if not log_path.exists():
        return {}

    content = log_path.read_text(encoding="utf-8", errors="replace")
    parsed: dict = {
        "run_id": None,
        "rounds": {},
        "local_epochs": [],
        "test_metrics": {},
    }

    # 1. Extract Run ID (taking the latest run in the log)
    run_matches = re.findall(r"runs/([a-f0-9]{32})", content)
    if run_matches:
        parsed["run_id"] = run_matches[-1]

    # 2. Extract every single local epoch for all clients across all rounds
    # e.g.: [Round 1 | Client 8] Epoch 1/10: loss=1.6712 | acc=0.4256 | f1_macro=0.2268
    local_epoch_pattern = re.compile(
        r"\[Round\s+(\d+)\s+\|\s+Client\s+(\d+)\]\s+Epoch\s+(\d+)/(\d+):\s+loss=([\d.]+)\s+\|\s+acc=([\d.]+)(?:\s+\|\s+f1_macro=([\d.]+))?"
    )
    for m in local_epoch_pattern.finditer(content):
        rnd = int(m.group(1))
        cid = int(m.group(2))
        ep = int(m.group(3))
        total_ep = int(m.group(4))
        loss = float(m.group(5))
        acc = float(m.group(6))
        f1 = float(m.group(7)) if m.group(7) is not None else None
        global_step = (rnd - 1) * total_ep + ep

        parsed["local_epochs"].append({
            "round": rnd,
            "client_id": cid,
            "local_epoch": ep,
            "total_local_epochs": total_ep,
            "global_epoch_step": global_step,
            "loss": loss,
            "accuracy": acc,
            "f1_macro": f1,
        })

    # 3. Extract Server Centralized Evaluation per round
    # e.g.: [Round 1] Server Centralized Eval: Loss=0.5952, Acc=0.7708, F1-Macro=0.5688
    server_eval_pattern = re.compile(
        r"\[Round\s+(\d+)\]\s+Server Centralized Eval:\s+Loss=([\d.]+),\s+Acc=([\d.]+),\s+F1-Macro=([\d.]+)"
    )
    for m in server_eval_pattern.finditer(content):
        rnd = int(m.group(1))
        parsed["rounds"][rnd] = {
            "val_loss": float(m.group(2)),
            "val_accuracy": float(m.group(3)),
            "val_f1_macro": float(m.group(4)),
        }

    # 4. Extract Final Test Set Results
    test_match = re.search(
        r"Accuracy:\s*([\d.]+)\s*\nPrecision Macro:\s*([\d.]+)\s*\nRecall Macro:\s*([\d.]+)\s*\nF1 Macro:\s*([\d.]+)",
        content,
    )
    if test_match:
        parsed["test_metrics"]["test_accuracy"] = float(test_match.group(1))
        parsed["test_metrics"]["test_precision_macro"] = float(test_match.group(2))
        parsed["test_metrics"]["test_recall_macro"] = float(test_match.group(3))
        parsed["test_metrics"]["test_f1_macro"] = float(test_match.group(4))

    lat_m = re.search(r"Single-sample inference latency:\s*([\d.]+)\s*ms", content)
    if lat_m:
        parsed["test_metrics"]["inference_latency_ms"] = float(lat_m.group(1))

    return parsed


def save_local_epochs_recap(local_epochs: list[dict], model_name: str) -> tuple[Path, Path]:
    """Save structured JSON and CSV recap files for all client local epochs."""
    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"fl_{model_name}_clients_local_epochs.json"
    csv_path = reports_dir / f"fl_{model_name}_clients_local_epochs.csv"

    # Save JSON
    json_path.write_text(json.dumps(local_epochs, indent=2))

    # Save CSV
    if local_epochs:
        fieldnames = list(local_epochs[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(local_epochs)

    return json_path, csv_path


def sync_fl_run(model_name: str = "hierarchical_transformer_gru", log_file: str = "log.txt") -> None:
    """Sync all local epoch metrics, round history, and artifacts directly to DagsHub MLflow."""
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
        print(f"\n[Sync] Found MLflow Run ID: {run_id}")
    else:
        print("\n[Sync] No existing run ID found; creating new MLflow run.")

    with mlflow.start_run(run_id=run_id, run_name=f"fl_{model_name}") as run:
        active_id = run.info.run_id

        # Update run_id_file
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(active_id)

        client = MlflowClient()
        now_ts = int(time.time() * 1000)

        # 1. Log Local Epoch Metrics (Up to 1,000 metrics logged in batches)
        local_epochs = parsed.get("local_epochs", [])
        if local_epochs:
            print(f"[Sync] Extracted {len(local_epochs):,} local epochs from {log_file} across all clients.")
            json_recap, csv_recap = save_local_epochs_recap(local_epochs, model_name)
            print(f"[Sync] Saved local epoch recaps to {json_recap} and {csv_recap}")

            # Build Metric objects for MLflow
            metric_objects = []
            # Group by global_epoch_step to compute step averages
            step_groups: dict = {}

            for item in local_epochs:
                cid = item["client_id"]
                step = item["global_epoch_step"]
                loss = item["loss"]
                acc = item["accuracy"]
                f1 = item["f1_macro"]

                step_groups.setdefault(step, {"losses": [], "accs": [], "f1s": []})
                step_groups[step]["losses"].append(loss)
                step_groups[step]["accs"].append(acc)

                metric_objects.append(Metric(f"client_{cid}_loss", loss, now_ts, step))
                metric_objects.append(Metric(f"client_{cid}_accuracy", acc, now_ts, step))
                if f1 is not None:
                    step_groups[step]["f1s"].append(f1)
                    metric_objects.append(Metric(f"client_{cid}_f1_macro", f1, now_ts, step))

            # Add step averages across participating clients
            for step, g in step_groups.items():
                if g["losses"]:
                    metric_objects.append(Metric("train_epoch_avg_loss", float(sum(g["losses"]) / len(g["losses"])), now_ts, step))
                if g["accs"]:
                    metric_objects.append(Metric("train_epoch_avg_accuracy", float(sum(g["accs"]) / len(g["accs"])), now_ts, step))
                if g["f1s"]:
                    metric_objects.append(Metric("train_epoch_avg_f1_macro", float(sum(g["f1s"]) / len(g["f1s"])), now_ts, step))

            # Upload metrics to MLflow in chunks of 500
            chunk_size = 500
            for i in range(0, len(metric_objects), chunk_size):
                client.log_batch(active_id, metrics=metric_objects[i : i + chunk_size])
            print(f"[Sync] Successfully logged {len(metric_objects):,} local epoch curve points to DagsHub MLflow.")

        # 2. Log Server Centralized Evaluation per Round
        rounds = parsed.get("rounds", {})
        if rounds:
            print(f"[Sync] Logging server centralized evaluation for {len(rounds)} rounds...")
            round_metric_objects = []
            for rnd, m_dict in sorted(rounds.items()):
                for m_name, m_val in m_dict.items():
                    round_metric_objects.append(Metric(m_name, float(m_val), now_ts, int(rnd)))
            client.log_batch(active_id, metrics=round_metric_objects)

        # 3. Log Test Evaluation Metrics
        test_m = parsed.get("test_metrics", {})
        if test_m:
            print(f"[Sync] Logging final test metrics: {test_m}")
            test_objects = [Metric(k, float(v), now_ts, 0) for k, v in test_m.items()]
            client.log_batch(active_id, metrics=test_objects)

        # 4. Ensure Model Report artifact exists
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

        # 5. Collect and upload all artifacts
        reports_dir = Path("reports/metrics")
        figures_dir = Path("reports/figures")
        artifacts_to_upload = [
            # Pre-training
            (model_report_path, "reports/model_summaries"),
            (processed_dir / "federated" / "partition_summary.json", "federated"),
            # Models
            (Path("checkpoints") / f"fl_{model_name}_best.keras", "models"),
            (Path("checkpoints") / f"fl_{model_name}_final.keras", "models"),
            (run_id_file, "checkpoints"),
            # Local Epoch Recaps
            (reports_dir / f"fl_{model_name}_clients_local_epochs.json", "metrics"),
            (reports_dir / f"fl_{model_name}_clients_local_epochs.csv", "metrics"),
            # Round Metrics
            (reports_dir / f"fl_{model_name}_metrics.json", "metrics"),
            (reports_dir / f"fl_{model_name}_best_metrics.json", "metrics"),
            # Evaluation Reports
            (reports_dir / f"fl_{model_name}_classification_report.json", "reports"),
            (reports_dir / f"fl_{model_name}_classification_report.txt", "reports"),
            # Confusion Matrix Figures
            (figures_dir / f"fl_{model_name}_test_confusion_matrix.png", "figures/confusion_matrices"),
            (figures_dir / f"fl_{model_name}_test_confusion_matrix_counts.png", "figures/confusion_matrices"),
            (figures_dir / f"fl_{model_name}_train_confusion_matrix.png", "figures/confusion_matrices"),
            (figures_dir / f"fl_{model_name}_val_confusion_matrix.png", "figures/confusion_matrices"),
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
        run_clean_uri = tracking_uri.removesuffix(".mlflow")
        print(f"[Sync] View run at: {run_clean_uri}.mlflow/#/experiments/18/runs/{active_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync FL artifacts to DagsHub")
    parser.add_argument("--model", type=str, default="hierarchical_transformer_gru", help="Model name")
    parser.add_argument("--log", type=str, default="log.txt", help="Log file path")
    args = parser.parse_args()

    sync_fl_run(model_name=args.model, log_file=args.log)
