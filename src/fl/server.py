"""Server-side Centralized Evaluation and FedAvg strategy configuration."""

import ctypes
import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import mlflow
import numpy as np
from tensorflow import keras

from fl.models import get_model_and_dataset
from utils.device_utils import configure_gpu_memory_growth

# Enable memory growth for server evaluation
configure_gpu_memory_growth()


def _trim_memory() -> None:
    """Force garbage collection and release heap pages back to OS kernel."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def get_eval_fn(
    model_name: str,
    val_path: str | Path,
    params: dict,
    best_model_path: str | Path,
    best_metrics_path: str | Path,
) -> Callable[[int, List[np.ndarray], Dict[str, Any]], Optional[Tuple[float, Dict[str, float]]]]:
    """
    Returns an evaluation function for centralized server-side evaluation.
    Evaluates global model weights on val.parquet after every communication round.
    """
    val_model, val_dataset = get_model_and_dataset(
        model_name=model_name,
        data_path=val_path,
        params=params,
        is_training=False,
    )
    val_tf_ds = val_dataset.to_tf_dataset()

    best_f1 = [float("-inf")]
    best_metrics_record = {}

    best_model_path = Path(best_model_path)
    best_metrics_path = Path(best_metrics_path)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    best_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        if server_round == 0:
            return None

        val_model.set_weights(parameters)
        eval_results = val_model.evaluate(val_tf_ds, verbose=0, return_dict=True)

        loss = float(eval_results.get("loss", 0.0))
        accuracy = float(eval_results.get("accuracy", 0.0))
        precision_macro = float(eval_results.get("precision_macro", 0.0))
        recall_macro = float(eval_results.get("recall_macro", 0.0))
        f1_macro = float(eval_results.get("f1_macro", 0.0))

        round_metrics = {
            "val_loss": loss,
            "val_accuracy": accuracy,
            "val_precision_macro": precision_macro,
            "val_recall_macro": recall_macro,
            "val_f1_macro": f1_macro,
        }

        # Log metrics to MLflow if active run exists
        if mlflow.active_run() is not None:
            mlflow.log_metrics(round_metrics, step=server_round)

        print(
            f"\n[Round {server_round}] Server Centralized Eval: "
            f"Loss={loss:.4f}, Acc={accuracy:.4f}, F1-Macro={f1_macro:.4f}"
        )

        # Track and save best model checkpoint
        if f1_macro > best_f1[0]:
            best_f1[0] = f1_macro
            best_metrics_record.clear()
            best_metrics_record.update({
                "best_round": server_round,
                "best_f1_macro": f1_macro,
                "metrics": round_metrics,
            })
            val_model.save(str(best_model_path))
            best_metrics_path.write_text(json.dumps(best_metrics_record, indent=2))
            print(f"[Round {server_round}] -> New best model saved to {best_model_path} (F1: {f1_macro:.4f})")

        _trim_memory()

        return loss, round_metrics

    return evaluate


def create_fit_metrics_aggregation_fn() -> Callable[[List[Tuple[int, Dict[str, Any]]]], Dict[str, Any]]:
    """Aggregate training metrics across all clients and print clean per-client summary."""

    def fit_metrics_agg(results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
        if not results:
            return {}

        total_samples = sum(num_samples for num_samples, _ in results)
        if total_samples == 0:
            return {}

        avg_loss = sum(num_samples * m.get("loss", 0.0) for num_samples, m in results) / total_samples
        avg_acc = sum(num_samples * m.get("accuracy", 0.0) for num_samples, m in results) / total_samples

        details = []
        for _, m in sorted(results, key=lambda x: int(x[1].get("client_id", 0)) if isinstance(x[1].get("client_id"), int) else 0):
            cid = m.get("client_id", "?")
            c_loss = m.get("loss", 0.0)
            c_acc = m.get("accuracy", 0.0)
            f1_val = m.get("f1_macro")
            f1_text = f", f1={float(f1_val):.4f}" if f1_val is not None else ""
            details.append(f"Client {cid} (loss={c_loss:.4f}, acc={c_acc:.4f}{f1_text})")

        print(
            f"\n[Round Summary - Clients Fit] Avg Loss={avg_loss:.4f} | Avg Acc={avg_acc:.4f}\n"
            f"  Participating: {', '.join(details)}",
            flush=True,
        )

        agg = {
            "train_avg_loss": avg_loss,
            "train_avg_accuracy": avg_acc,
        }
        if any("f1_macro" in m for _, m in results):
            avg_f1 = sum(num_samples * float(m.get("f1_macro", 0.0)) for num_samples, m in results) / total_samples
            agg["train_avg_f1_macro"] = avg_f1

        return agg

    return fit_metrics_agg


def create_fedavg_strategy(
    initial_parameters: List[np.ndarray],
    eval_fn: Callable,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
) -> fl.server.strategy.FedAvg:
    """Instantiate standard FedAvg strategy with server evaluation."""
    initial_ndarrays = fl.common.ndarrays_to_parameters(initial_parameters)

    return fl.server.strategy.FedAvg(
        fraction_fit=float(fraction_fit),
        fraction_evaluate=0.0,  # Using centralized server-side evaluation
        min_fit_clients=int(min_fit_clients),
        min_evaluate_clients=0,
        min_available_clients=int(min_available_clients),
        evaluate_fn=eval_fn,
        fit_metrics_aggregation_fn=create_fit_metrics_aggregation_fn(),
        initial_parameters=initial_ndarrays,
        on_fit_config_fn=lambda rnd: {"round": rnd},
    )
