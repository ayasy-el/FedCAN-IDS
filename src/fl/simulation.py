"""Flower Federated Learning Simulation Runner for FedCAN-IDS."""

import argparse
import gc
import json
import os
from pathlib import Path

# Prevent Ray from prematurely killing workers on transient RAM spikes
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import flwr as fl
import mlflow
import tensorflow as tf

from utils.device_utils import configure_gpu_memory_growth

# Enable incremental GPU memory allocation before TensorFlow initializes CUDA
configure_gpu_memory_growth()

from fl.client import FedCanClient
from fl.models import get_model_and_dataset
from fl.partition import partition_windows
from fl.server import create_fedavg_strategy, get_eval_fn
from utils.mlflow_utils import init_dagshub, save_run_id
from utils.params import load_params


def run_federated_simulation(model_name: str = "hierarchical_transformer_gru", params: dict | None = None) -> None:
    """Run Flower Federated Learning simulation using configured parameters."""
    if params is None:
        params = load_params()

    dataset_params = params["dataset"]
    fl_params = params.get("federated", {})
    mlflow_params = params.get("mlflow", {})
    split_params = params.get("split", {})

    num_clients = int(fl_params.get("num_clients", 5))
    num_rounds = int(fl_params.get("num_rounds", 20))
    local_epochs = int(fl_params.get("local_epochs", 2))
    batch_size = int(fl_params.get("batch_size", 32))
    fraction_fit = float(fl_params.get("fraction_fit", 1.0))

    # Initialize DagsHub / MLflow
    init_dagshub()
    tracking_uri = mlflow_params.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = mlflow_params.get("experiment_fl", "federated_can_ids")
    mlflow.set_experiment(experiment_name)

    # Output paths
    processed_dir = Path(dataset_params["processed_dir"])
    federated_dir = processed_dir / "federated"
    val_path = processed_dir / "val.parquet"
    best_model_path = Path("checkpoints") / f"fl_{model_name}_best.keras"
    final_model_path = Path("checkpoints") / f"fl_{model_name}_final.keras"
    best_metrics_path = Path("reports/metrics") / f"fl_{model_name}_best_metrics.json"
    summary_report_path = Path("reports/metrics") / f"fl_{model_name}_metrics.json"
    run_id_path = Path("checkpoints") / f"fl_{model_name}_mlflow_run_id.txt"

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    best_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_id_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure federated partition files exist
    client_files = [federated_dir / f"client_{i}.parquet" for i in range(num_clients)]
    if not all(p.exists() for p in client_files):
        print(f"Federated partitions missing in {federated_dir}. Generating now...")
        train_path = processed_dir / "train.parquet"
        partition_windows(
            train_path=train_path,
            output_dir=federated_dir,
            num_clients=num_clients,
            seed=int(split_params.get("random_seed", 42)),
        )

    # 1. Initialize Base Model to extract initial parameters (on CPU to save VRAM for Ray workers)
    with tf.device("/CPU:0"):
        base_model, _ = get_model_and_dataset(
            model_name=model_name,
            data_path=val_path,
            params=params,
            is_training=False,
        )
    initial_weights = base_model.get_weights()

    # 2. Server-side centralized evaluation function
    eval_fn = get_eval_fn(
        model_name=model_name,
        val_path=val_path,
        params=params,
        best_model_path=best_model_path,
        best_metrics_path=best_metrics_path,
    )

    # 3. Strategy Setup (FedAvg)
    min_fit = max(1, int(round(num_clients * fraction_fit)))
    strategy = create_fedavg_strategy(
        initial_parameters=initial_weights,
        eval_fn=eval_fn,
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit,
        min_available_clients=num_clients,
    )

    # 4. Client Factory Function for Flower Simulation (cached per cid to eliminate RAM leaks)
    client_cache: dict[int, FedCanClient] = {}

    def client_fn(context: fl.common.Context) -> fl.client.Client:
        partition_id = context.node_config.get("partition-id") if hasattr(context, "node_config") and context.node_config else None
        if partition_id is not None:
            cid = int(partition_id)
        else:
            cid = int(context.node_id) % num_clients

        if cid not in client_cache:
            client_parquet = client_files[cid]
            client_cache[cid] = FedCanClient(
                client_id=cid,
                model_name=model_name,
                client_path=client_parquet,
                params=params,
                local_epochs=local_epochs,
                batch_size=batch_size,
            )
        return client_cache[cid].to_client()

    # 5. Device detection for Ray simulation resources
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Allocate 1.0 GPU per virtual client to run sequentially and prevent VRAM OOM
        client_resources = {"num_cpus": 1, "num_gpus": 1.0}
        print(f"[Simulation] GPU detected ({len(gpus)} device(s)): running clients sequentially on GPU")
    else:
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}
        print("[Simulation] Running clients on CPU")

    with mlflow.start_run(run_name=f"fl_{model_name}") as run:
        save_run_id(run.info.run_id, str(run_id_path))
        mlflow.log_params({
            "model": model_name,
            "fl_strategy": "FedAvg",
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "fraction_fit": fraction_fit,
        })

        print(f"\n{'='*70}\nStarting Flower FL Simulation: {model_name.upper()} ({num_clients} clients, {num_rounds} rounds)\n{'='*70}")

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )

        # Save final global model
        base_model.save(str(final_model_path))
        print(f"Final global model saved to {final_model_path}")

        # Save summary report
        summary = {
            "model": model_name,
            "strategy": "FedAvg",
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "losses_distributed": [float(x[1]) for x in history.losses_distributed] if history.losses_distributed else [],
            "losses_centralized": [float(x[1]) for x in history.losses_centralized] if history.losses_centralized else [],
        }
        summary_report_path.write_text(json.dumps(summary, indent=2))
        print(f"Federated training summary written to {summary_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument(
        "--model",
        type=str,
        default="hierarchical_transformer_gru",
        help="Model architecture: 'hierarchical_transformer_gru', 'cnn_lstm', 'transformer_gru', 'mlp_window'",
    )
    args = parser.parse_args()

    run_federated_simulation(model_name=args.model)
