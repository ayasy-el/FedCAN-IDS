"""Model and dataset factory for Federated Learning."""

from pathlib import Path
from typing import Any, Tuple

from tensorflow import keras

from utils.metrics import MacroF1Score, MacroPrecision, MacroRecall
from utils.params import load_params


def get_model_and_dataset(
    model_name: str,
    data_path: str | Path,
    params: dict,
    is_training: bool = True,
    batch_size: int | None = None,
) -> Tuple[keras.Model, Any]:
    """
    Factory function to instantiate the requested model and its corresponding dataset.
    """
    dataset_cfg = params["dataset"]
    split_cfg = params["split"]
    window_size = int(split_cfg.get("window_size", 64))
    source_path = dataset_cfg.get("prepared_path")
    model_params = params["model"].get(model_name, {})
    training_params = params["training"].get(model_name, {})

    from data.task import validate_experiment

    task = validate_experiment(params, model_name)
    num_classes = task["num_classes"]

    bs = batch_size or int(training_params.get("batch_size", 32))
    lr = float(training_params.get("learning_rate", 1e-3))

    if model_name == "cnn_lstm":
        from data.cnn_lstm_dataset import CNNLSTMDataset
        from model.cnn_lstm import build_cnn_lstm

        num_features = int(model_params.get("num_features", 5))
        dataset = CNNLSTMDataset(
            data_path,
            window_size=window_size,
            rolling_w=int(model_params.get("rolling_w", 64)),
            num_features=num_features,
            batch_size=bs,
            shuffle=is_training,
            random_seed=int(split_cfg.get("random_seed", 42)),
            source_path=source_path,
            label_rule=model_params.get("label_rule", "any_attack"),
        )
        model = build_cnn_lstm(
            window_size=window_size,
            num_features=num_features,
            num_classes=num_classes,
            cnn_filters=int(model_params.get("cnn_filters", 16)),
            cnn_kernel_size=int(model_params.get("cnn_kernel_size", 3)),
            dropout=float(model_params.get("dropout", 0.2)),
            lstm_hidden_size=int(model_params.get("lstm_hidden_size", 32)),
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                MacroPrecision(num_classes),
                MacroRecall(num_classes),
                MacroF1Score(num_classes),
            ],
        )
        return model, dataset

    elif model_name == "transformer_gru":
        from data.transformer_gru_dataset import TransformerGRUDataset
        from model.transformer_gru import build_transformer_gru

        num_features = int(model_params.get("num_features", 5))
        dataset = TransformerGRUDataset(
            data_path,
            window_size=window_size,
            rolling_w=int(model_params.get("rolling_w", 50)),
            num_features=num_features,
            batch_size=bs,
            shuffle=is_training,
            random_seed=int(split_cfg.get("random_seed", 42)),
            source_path=source_path,
            label_rule=model_params.get("label_rule", "any_attack"),
        )
        model = build_transformer_gru(
            window_size=window_size,
            num_features=num_features,
            num_classes=num_classes,
            d_model=int(model_params.get("d_model", 16)),
            num_heads=int(model_params.get("num_heads", 2)),
            dim_feedforward=int(model_params.get("dim_feedforward", 32)),
            dropout=float(model_params.get("dropout", 0.2)),
            gru_hidden_size=int(model_params.get("gru_hidden_size", 32)),
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                MacroPrecision(num_classes),
                MacroRecall(num_classes),
                MacroF1Score(num_classes),
            ],
        )
        return model, dataset

    elif model_name == "hierarchical_transformer_gru":
        from data.transformer_gru_dataset import TransformerGRUDataset
        from model.hierarchical_transformer_gru import build_hierarchical_transformer_gru

        num_features = int(model_params.get("num_features", 5))
        dataset = TransformerGRUDataset(
            data_path,
            window_size=window_size,
            rolling_w=int(model_params.get("rolling_w", 50)),
            num_features=num_features,
            batch_size=bs,
            shuffle=is_training,
            random_seed=int(split_cfg.get("random_seed", 42)),
            source_path=source_path,
            label_rule=model_params.get("label_rule", "last_frame"),
        )
        model = build_hierarchical_transformer_gru(
            window_size=window_size,
            num_features=num_features,
            num_classes=num_classes,
            d_model=int(model_params.get("d_model", 16)),
            num_heads=int(model_params.get("num_heads", 2)),
            dim_feedforward=int(model_params.get("dim_feedforward", 32)),
            dropout=float(model_params.get("dropout", 0.2)),
            gru_hidden_size=int(model_params.get("gru_hidden_size", 32)),
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                MacroPrecision(num_classes),
                MacroRecall(num_classes),
                MacroF1Score(num_classes),
            ],
        )
        return model, dataset

    elif model_name == "mlp_window":
        from data.mlp_window_dataset import MLPWindowDataset
        from model.mlp_window import build_mlp_window

        dataset = MLPWindowDataset(
            data_path,
            normalize_stats_path=f"checkpoints/{model_name}_norm_stats.json",
            fit_normalize_stats=is_training,
            can_id_bits=int(model_params.get("can_id_bits", 11)),
            batch_size=bs,
            shuffle=is_training,
            source_path=source_path,
            window_size=window_size,
        )
        model = build_mlp_window(
            dataset.input_dim,
            num_classes,
            model_params.get("hidden_dims", [32, 16]),
            model_params.get("dropout", False),
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                MacroPrecision(num_classes),
                MacroRecall(num_classes),
                MacroF1Score(num_classes),
            ],
        )
        return model, dataset

    else:
        raise ValueError(
            f"Unsupported Federated Learning model: {model_name!r}. "
            "Supported: 'hierarchical_transformer_gru', 'cnn_lstm', 'transformer_gru', 'mlp_window'."
        )
