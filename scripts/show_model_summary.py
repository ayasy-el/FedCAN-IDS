"""Show configured TensorFlow model summaries without training or MLflow."""

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from data.task import MODEL_CONTRACTS, label_schema
from model.can_ae_transformer import (
    build_can_ae_transformer,
    parameter_counts as can_ae_transformer_parameter_counts,
)
from model.can_bigrubert import build_can_bigrubert, parameter_counts
from model.mlp import build_mlp
from model.mlp_window import build_mlp_window
from model.tct_ids import (
    build_tct_ids,
    parameter_counts as tct_ids_parameter_counts,
)


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
    if model_name == "can_ae_transformer":
        from data.can_ae_transformer_dataset import CANAeTransformerDataset

        return CANAeTransformerDataset(
            train_path,
            window_size=int(split["window_size"]),
            d_model=int(model_params["d_model"]),
            granularity=float(model_params.get("granularity", 1e-8)),
            max_time_position=int(model_params.get("max_time_position", 10000)),
            batch_size=training["batch_size"],
            shuffle=False,
            random_seed=split["random_seed"],
            source_path=PROJECT_ROOT / dataset["prepared_path"],
        )
    if model_name == "tct_ids":
        from data.tct_ids_dataset import TCTIDSDataset

        return TCTIDSDataset(
            train_path,
            window_size=int(split["window_size"]),
            d_model=int(model_params.get("d_model", 10)),
            alpha=float(model_params.get("alpha_smooth", 1e-7)),
            batch_size=training["batch_size"],
            shuffle=False,
            random_seed=split["random_seed"],
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


def show_can_ae_transformer_summary(params, schema, split):
    model_params = params["model"]["can_ae_transformer"]
    window_size = int(split["window_size"])
    model = build_can_ae_transformer(
        window_size=window_size,
        d_model=int(model_params["d_model"]),
        num_heads=int(model_params["num_heads"]),
        num_layers=int(model_params["num_layers"]),
        dim_feedforward=int(model_params["dim_feedforward"]),
        dropout=float(model_params["dropout"]),
        num_classes=schema["num_classes"],
        batch_size=_batch_size(params, "can_ae_transformer"),
    )
    print("\n=== CAN-AE-Transformer window classifier ===")
    _print_contract("can_ae_transformer", split)
    print(f"Window size: {window_size}")
    print(f"Stride: {split['stride']}")
    print(f"Embedding dimension (d_model): {model_params['d_model']}")
    print(f"Attention heads: {model_params['num_heads']}")
    print(f"Encoder layers: {model_params['num_layers']}")
    print(f"Feedforward dimension: {model_params['dim_feedforward']}")
    print(f"Smooth factor (granularity): {model_params.get('granularity', 1e-8)}")
    print(f"Dropout: {model_params['dropout']}")
    _print_batch_info(
        params,
        "can_ae_transformer",
        (window_size, 12),
        schema["num_classes"],
    )
    model.summary()
    for name, value in can_ae_transformer_parameter_counts(model).items():
        print(f"{name}: {value:,}")
    return model


def show_tct_ids_summary(params, schema, split):
    model_params = params["model"]["tct_ids"]
    window_size = int(split["window_size"])
    model = build_tct_ids(
        window_size=window_size,
        d_model=int(model_params.get("d_model", 10)),
        num_heads=int(model_params.get("num_heads", 1)),
        num_layers=int(model_params.get("num_layers", 3)),
        dim_feedforward=int(model_params.get("dim_feedforward", 40)),
        mlp_hidden_dim=int(model_params.get("mlp_hidden_dim", 40)),
        tcn_filters=int(model_params.get("tcn_filters", 100)),
        tcn_kernel_size=int(model_params.get("tcn_kernel_size", 2)),
        tcn_dilations=model_params.get("tcn_dilations", [1, 2, 4]),
        use_weight_norm=bool(model_params.get("use_weight_norm", True)),
        dropout=float(model_params.get("dropout", 0.1)),
        num_classes=schema["num_classes"],
        fusion=model_params.get("fusion", "concat"),
        pooling=model_params.get("pooling", "last"),
        batch_size=_batch_size(params, "tct_ids"),
    )
    print("\n=== TCT-IDS window classifier ===")
    _print_contract("tct_ids", split)
    print(f"Window size: {window_size}")
    print(f"Stride: {split['stride']}")
    print(f"Embedding dimension (d_model): {model_params.get('d_model', 10)}")
    print(f"Attention heads: {model_params.get('num_heads', 1)}")
    print(f"Encoder layers: {model_params.get('num_layers', 3)}")
    print(f"Feedforward dimension: {model_params.get('dim_feedforward', 40)}")
    print(f"TCN filters: {model_params.get('tcn_filters', 100)}")
    print(f"TCN kernel size: {model_params.get('tcn_kernel_size', 2)}")
    print(f"TCN dilations: {model_params.get('tcn_dilations', [1, 2, 4])}")
    print(f"Weight normalization: {model_params.get('use_weight_norm', True)}")
    print(f"Fusion method: {model_params.get('fusion', 'concat')}")
    print(f"Pooling method: {model_params.get('pooling', 'last')}")
    print(f"Dropout: {model_params.get('dropout', 0.1)}")
    _print_batch_info(
        params,
        "tct_ids",
        (window_size, 12),
        schema["num_classes"],
    )
    model.summary()
    for name, value in tct_ids_parameter_counts(model).items():
        print(f"{name}: {value:,}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Show configured model summaries without training or MLflow."
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=("mlp", "mlp_window", "can_bigrubert", "can_ae_transformer", "tct_ids", "all"),
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
    if args.model in ("can_ae_transformer", "all"):
        model = show_can_ae_transformer_summary(params, schema, split)
        if args.estimate_time:
            try:
                print_training_estimate(
                    params, "can_ae_transformer", model, args.benchmark_steps
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"Time estimate skipped for can_ae_transformer: {exc}")
    if args.model in ("tct_ids", "all"):
        model = show_tct_ids_summary(params, schema, split)
        if args.estimate_time:
            try:
                print_training_estimate(
                    params, "tct_ids", model, args.benchmark_steps
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"Time estimate skipped for tct_ids: {exc}")


if __name__ == "__main__":
    main()
