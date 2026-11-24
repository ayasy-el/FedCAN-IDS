"""Stratified Random Window partitioning for Federated Learning."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from utils.params import load_params


def partition_windows(
    train_path: str | Path,
    output_dir: str | Path,
    num_clients: int = 5,
    seed: int = 42,
) -> list[Path]:
    """
    Partition train.parquet into N client datasets using stratified random partitioning.
    Each client receives an equal, disjoint, and representative share of every class's windows.
    """
    train_path = Path(train_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    df = pl.read_parquet(train_path)
    rng = np.random.default_rng(seed)

    client_dfs = [[] for _ in range(num_clients)]
    summary_stats = {
        "num_clients": num_clients,
        "total_windows": len(df),
        "clients": {},
    }

    # Group by label to ensure exact class distribution across all clients
    unique_labels = sorted(int(l) for l in df["label"].unique().to_list())
    for label in unique_labels:
        label_df = df.filter(pl.col("label") == label)
        indices = np.arange(len(label_df))
        rng.shuffle(indices)

        # Distribute indices into num_clients chunks
        chunks = np.array_split(indices, num_clients)
        for client_id, chunk in enumerate(chunks):
            if len(chunk) > 0:
                client_dfs[client_id].append(label_df[chunk])

    output_paths = []
    for client_id in range(num_clients):
        if client_dfs[client_id]:
            client_data = pl.concat(client_dfs[client_id]).sort(["session_id", "start_row_id"])
        else:
            client_data = df.head(0)

        out_path = output_dir / f"client_{client_id}.parquet"
        client_data.write_parquet(out_path, compression="zstd")
        output_paths.append(out_path)

        class_counts = {
            int(lbl): int(cnt)
            for lbl, cnt in zip(*np.unique(client_data["label"].to_numpy(), return_counts=True))
        }
        summary_stats["clients"][f"client_{client_id}"] = {
            "total_windows": len(client_data),
            "class_distribution": class_counts,
            "path": str(out_path),
        }
        print(f"Client {client_id}: {len(client_data):,} windows saved to {out_path}")

    summary_path = output_dir / "partition_summary.json"
    summary_path.write_text(json.dumps(summary_stats, indent=2))
    print(f"Partition summary written to {summary_path}")

    return output_paths


def run(params: dict | None = None) -> list[Path]:
    if params is None:
        params = load_params()

    dataset_params = params["dataset"]
    fl_params = params.get("federated", {})
    split_params = params.get("split", {})

    train_path = Path(dataset_params["processed_dir"]) / "train.parquet"
    output_dir = Path(dataset_params["processed_dir"]) / "federated"
    num_clients = int(fl_params.get("num_clients", 5))
    seed = int(split_params.get("random_seed", 42))

    return partition_windows(
        train_path=train_path,
        output_dir=output_dir,
        num_clients=num_clients,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition windows for Federated Learning")
    parser.add_argument("--num-clients", type=int, default=None, help="Number of clients")
    args = parser.parse_args()

    params = load_params()
    if args.num_clients is not None:
        params.setdefault("federated", {})["num_clients"] = args.num_clients

    run(params)
