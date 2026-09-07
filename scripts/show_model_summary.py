"""Show model architecture summaries without training or MLflow/DagsHub init."""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model.mlp import build_mlp
from model.mlp_window import build_mlp_window


def load_project_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open() as params_file:
        return yaml.safe_load(params_file)


def show_single_frame_summary(params: dict) -> None:
    model_params = params["model"]["mlp"]
    feature_dim = model_params["can_id_bits"] + 11
    model = build_mlp(feature_dim, model_params["num_classes"])
    print("\n=== Single-frame MLP ===")
    print(f"Input features: {feature_dim}")
    model.summary()
    print(f"Total parameters: {model.count_params():,}")


def show_window_summary(params: dict) -> None:
    model_params = params["model"]["mlp_window"]
    frame_feature_dim = model_params["can_id_bits"] + 11
    input_dim = frame_feature_dim * model_params["window_size"]
    model = build_mlp_window(
        input_dim=input_dim,
        n_classes=model_params["num_classes"],
        hidden_dims=model_params["hidden_dims"],
        dropout=model_params["dropout"],
    )
    print("\n=== Sliding-window MLP ===")
    print(f"Frame features: {frame_feature_dim}")
    print(f"Window size: {model_params['window_size']}")
    print(f"Stride: {model_params['stride']}")
    print(f"Flattened input features: {input_dim}")
    print(f"Hidden dims: {model_params['hidden_dims']}")
    model.summary()
    print(f"Total parameters: {model.count_params():,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show configured model summaries without training or MLflow."
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=("mlp", "mlp_window", "all"),
        default="all",
        help="Model summary to show (default: all).",
    )
    args = parser.parse_args()
    params = load_project_params()

    if args.model in ("mlp", "all"):
        show_single_frame_summary(params)
    if args.model in ("mlp_window", "all"):
        show_window_summary(params)


if __name__ == "__main__":
    main()
