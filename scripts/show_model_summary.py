"""Show configured TensorFlow model summaries without training or MLflow."""

import argparse
import math
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.task import MODEL_CONTRACTS, label_schema
from model.can_bigrubert import build_can_bigrubert, parameter_counts
from model.mlp import build_mlp
from model.mlp_window import build_mlp_window


def load_project_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open() as params_file:
        return yaml.safe_load(params_file)


def _task_info(params):
    schema_name = params.get("prepare", {}).get("label_schema", "five_class")
    schema = label_schema(schema_name, params.get("dataset", {}).get("variant"))
    split = params.get("split", {})
    print(f"Label schema: {schema_name} ({schema['num_classes']} classes)")
    print(f"Configured split unit: {split.get('unit', 'frame')}")
    return schema, split


def _print_contract(model_name, split):
    required = MODEL_CONTRACTS[model_name]["unit"]
    configured = split.get("unit", "frame")
    status = (
        "compatible"
        if required == configured
        else "summary only; incompatible for current split"
    )
    print(f"Data contract: {required} ({status})")


def _batch_size(params, model_name):
    return (
        int(params.get("training", {}).get(model_name, {}).get("batch_size", 0)) or None
    )


def _print_batch_info(params, model_name, input_shape, output_classes):
    batch_size = _batch_size(params, model_name)
    if batch_size is None:
        print("Configured batch size: not set (dynamic)")
        return None
    print(f"Configured batch size: {batch_size}")
    print(f"Concrete input shape: ({batch_size}, {', '.join(map(str, input_shape))})")
    print(f"Concrete output shape: ({batch_size}, {output_classes})")
    return batch_size


def _format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _training_dataset(params, model_name):
    dataset = params["dataset"]
    split = params["split"]
    model_params = params["model"][model_name]
    training = params["training"][model_name]
    train_path = str(PROJECT_ROOT / dataset["processed_dir"] / "train.parquet")

    if model_name == "mlp":
        from data.mlp_dataset import MLPCANDataset

        return MLPCANDataset(
            train_path,
            "checkpoints/mlp_norm_stats.json",
            False,
            model_params["can_id_bits"],
            training["batch_size"],
            False,
        )
    if model_name == "mlp_window":
        from data.mlp_window_dataset import MLPWindowDataset

        return MLPWindowDataset(
            train_path,
            "checkpoints/mlp_window_norm_stats.json",
            False,
            model_params["can_id_bits"],
            training["batch_size"],
            False,
            source_path=PROJECT_ROOT / dataset["prepared_path"],
        )
    if model_name == "can_bigrubert":
        from data.can_bigrubert_dataset import CANBiGRUBERTDataset

        return CANBiGRUBERTDataset(
            train_path,
            model_params["tokenizer_checkpoint"],
            int(split["window_size"]),
            int(model_params["max_length"]),
            training["batch_size"],
            False,
            split["random_seed"],
            source_path=PROJECT_ROOT / dataset["prepared_path"],
        )
    raise ValueError(f"Unsupported model: {model_name}")


def print_training_estimate(params, model_name, model, benchmark_steps):
    from tensorflow import keras

    training = params.get("training", {}).get(model_name, {})
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    dataset = _training_dataset(params, model_name)
    samples = len(dataset.y)
    steps_per_epoch = math.ceil(samples / batch_size)
    optimizer = keras.optimizers.Adam(training["learning_rate"])
    if model_name == "can_bigrubert":
        optimizer = keras.optimizers.AdamW(
            learning_rate=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    batches = iter(dataset.to_tf_dataset())
    try:
        warmup_batch = next(batches)
        model.train_on_batch(*warmup_batch)
        elapsed_steps = []
        for _ in range(benchmark_steps):
            start = time.perf_counter()
            batch = next(batches)
            model.train_on_batch(*batch)
            elapsed_steps.append(time.perf_counter() - start)
    except StopIteration as exc:
        raise ValueError(
            "Training dataset has fewer batches than benchmark steps"
        ) from exc
    seconds_per_step = sum(elapsed_steps) / len(elapsed_steps)
    seconds_per_epoch = steps_per_epoch * seconds_per_step
    total_seconds = seconds_per_epoch * epochs
    print("Training time estimate:")
    print(f"  Training samples: {samples:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Benchmark steps: {benchmark_steps} (+ 1 warm-up step)")
    print(f"  Measured average seconds per step: {seconds_per_step:.3f}")
    print(f"  Estimated one epoch: {_format_duration(seconds_per_epoch)}")
    print(f"  Configured epochs: {epochs}")
    print(f"  Estimated training total: {_format_duration(total_seconds)}")


def show_mlp_summary(params, schema, split):
    model_params = params["model"]["mlp"]
    input_dim = int(model_params["can_id_bits"]) + 11
    batch_size = _batch_size(params, "mlp")
    model = build_mlp(input_dim, schema["num_classes"], batch_size=batch_size)
    print("\n=== MLP frame classifier ===")
    _print_contract("mlp", split)
    print(f"Input features: {input_dim}")
    print(f"CAN ID bits: {model_params['can_id_bits']}")
    _print_batch_info(params, "mlp", (input_dim,), schema["num_classes"])
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
    return model


def show_mlp_window_summary(params, schema, split):
    model_params = params["model"]["mlp_window"]
    window_size = int(split["window_size"])
    frame_feature_dim = int(model_params["can_id_bits"]) + 11
    input_dim = frame_feature_dim * window_size
    model = build_mlp_window(
        input_dim=input_dim,
        n_classes=schema["num_classes"],
        hidden_dims=model_params["hidden_dims"],
        dropout=model_params["dropout"],
        batch_size=_batch_size(params, "mlp_window"),
    )
    print("\n=== MLP window classifier ===")
    _print_contract("mlp_window", split)
    print(f"Frame features: {frame_feature_dim}")
    print(f"Window size: {window_size}")
    print(f"Stride: {split['stride']}")
    print(f"Flattened input features: {input_dim}")
    print(f"Hidden dims: {model_params['hidden_dims']}")
    print(f"Dropout: {model_params['dropout']}")
    _print_batch_info(params, "mlp_window", (input_dim,), schema["num_classes"])
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
    return model


def show_can_bigrubert_summary(params, schema, split):
    model_params = params["model"]["can_bigrubert"]
    window_size = int(split["window_size"])
    model = build_can_bigrubert(
        window_size=window_size,
        max_length=int(model_params["max_length"]),
        bert_checkpoint=model_params["bert_checkpoint"],
        bigru_hidden_size=int(model_params["bigru_hidden_size"]),
        dropout=model_params["dropout"],
        num_classes=schema["num_classes"],
        batch_size=_batch_size(params, "can_bigrubert"),
    )
    print("\n=== CAN-BiGRUBERT window classifier ===")
    _print_contract("can_bigrubert", split)
    print(f"Window size: {window_size}")
    print(f"Stride: {split['stride']}")
    print(f"Max token length: {model_params['max_length']}")
    print(f"BERT checkpoint: {model_params['bert_checkpoint']}")
    print(f"BiGRU hidden size: {model_params['bigru_hidden_size']}")
    print(f"Dropout: {model_params['dropout']}")
    _print_batch_info(
        params,
        "can_bigrubert",
        (window_size, int(model_params["max_length"])),
        schema["num_classes"],
    )
    model.summary()
    for name, value in parameter_counts(model).items():
        print(f"{name}: {value:,}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Show configured model summaries without training or MLflow."
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=("mlp", "mlp_window", "can_bigrubert", "all"),
        default="all",
        help="Model summary to show (default: all).",
    )
    parser.add_argument(
        "--estimate-time",
        action="store_true",
        help="Benchmark a few real training steps and estimate epoch/total time.",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=3,
        metavar="N",
        help="Measured steps after one warm-up step (default: 3).",
    )
    args = parser.parse_args()
    if args.benchmark_steps < 1:
        parser.error("--benchmark-steps must be at least 1")
    params = load_project_params()
    schema, split = _task_info(params)

    if args.model in ("mlp", "all"):
        model = show_mlp_summary(params, schema, split)
        if args.estimate_time:
            try:
                print_training_estimate(params, "mlp", model, args.benchmark_steps)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Time estimate skipped for mlp: {exc}")
    if args.model in ("mlp_window", "all"):
        model = show_mlp_window_summary(params, schema, split)
        if args.estimate_time:
            try:
                print_training_estimate(
                    params, "mlp_window", model, args.benchmark_steps
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"Time estimate skipped for mlp_window: {exc}")
    if args.model in ("can_bigrubert", "all"):
        model = show_can_bigrubert_summary(params, schema, split)
        if args.estimate_time:
            try:
                print_training_estimate(
                    params, "can_bigrubert", model, args.benchmark_steps
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"Time estimate skipped for can_bigrubert: {exc}")


if __name__ == "__main__":
    main()
