"""Unit tests for Federated Learning module (Flower)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from fl.client import FedCanClient
from fl.partition import partition_windows
from fl.server import create_fedavg_strategy, get_eval_fn


def test_partition_windows_stratified_and_disjoint(tmp_path):
    train_file = tmp_path / "train.parquet"
    output_dir = tmp_path / "federated"

    # Create dummy train dataset with 3 classes
    n_per_class = 60
    records = []
    for lbl in (0, 1, 2):
        for i in range(n_per_class):
            records.append({
                "window_id": f"sess_1:{lbl}_{i}",
                "session_id": "sess_1",
                "start_row_id": int(lbl * 1000 + i),
                "window_size": 15,
                "label": int(lbl),
            })
    df = pl.DataFrame(records)
    df.write_parquet(train_file)

    num_clients = 3
    client_paths = partition_windows(
        train_path=train_file,
        output_dir=output_dir,
        num_clients=num_clients,
        seed=42,
    )

    assert len(client_paths) == num_clients
    all_window_ids = set()

    for p in client_paths:
        assert p.exists()
        cdf = pl.read_parquet(p)
        assert len(cdf) == 60  # 180 total / 3 clients = 60
        # Check all 3 classes are present
        assert set(cdf["label"].unique().to_list()) == {0, 1, 2}
        # Check for duplicate window_id
        c_wids = set(cdf["window_id"].to_list())
        assert len(c_wids & all_window_ids) == 0
        all_window_ids.update(c_wids)

    assert len(all_window_ids) == 180
    summary_file = output_dir / "partition_summary.json"
    assert summary_file.exists()


def test_fedcan_client_lifecycle():
    mock_model = MagicMock()
    mock_weights = [np.zeros((10, 2)), np.zeros(2)]
    mock_model.get_weights.return_value = mock_weights

    mock_history = MagicMock()
    mock_history.history = {"loss": [0.25], "accuracy": [0.95]}
    mock_model.fit.return_value = mock_history
    mock_model.evaluate.return_value = (0.20, 0.96)

    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 100
    mock_dataset.to_tf_dataset.return_value = None

    with patch("fl.client.get_model_and_dataset", return_value=(mock_model, mock_dataset)):
        client = FedCanClient(
            client_id=0,
            model_name="cnn_lstm",
            client_path="dummy.parquet",
            params={},
            local_epochs=2,
            batch_size=32,
        )

        # 1. get_parameters
        params = client.get_parameters({})
        assert len(params) == 2

        # 2. fit
        new_params, num_samples, metrics = client.fit(mock_weights, {})
        assert num_samples == 100
        assert metrics["accuracy"] == 0.95
        assert metrics["loss"] == 0.25
        mock_model.fit.assert_called_once()

        # 3. evaluate
        loss, eval_samples, eval_metrics = client.evaluate(mock_weights, {})
        assert loss == 0.20
        assert eval_metrics["accuracy"] == 0.96


def test_fedavg_strategy_creation():
    mock_eval_fn = MagicMock()
    initial_weights = [np.ones((5, 2))]

    strategy = create_fedavg_strategy(
        initial_parameters=initial_weights,
        eval_fn=mock_eval_fn,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=3,
    )

    assert strategy.fraction_fit == 1.0
    assert strategy.min_fit_clients == 2
    assert strategy.min_available_clients == 3
    assert strategy.evaluate_fn == mock_eval_fn


def test_models_factory_hierarchical_transformer_gru():
    from fl.models import get_model_and_dataset

    mock_model = MagicMock()
    mock_dataset = MagicMock()

    with patch("data.task.validate_experiment", return_value={"num_classes": 5}):
        with patch("model.hierarchical_transformer_gru.build_hierarchical_transformer_gru", return_value=mock_model):
            with patch("data.transformer_gru_dataset.TransformerGRUDataset", return_value=mock_dataset):
                params = {
                    "dataset": {"prepared_path": "dummy_prep.parquet"},
                    "split": {"window_size": 20, "random_seed": 42},
                    "model": {"hierarchical_transformer_gru": {"d_model": 16, "num_heads": 2, "dim_feedforward": 32, "gru_hidden_size": 32}},
                    "training": {"hierarchical_transformer_gru": {"learning_rate": 0.001, "batch_size": 32}},
                }
                model, dataset = get_model_and_dataset("hierarchical_transformer_gru", "dummy_train.parquet", params)
                assert model == mock_model
                assert dataset == mock_dataset


def test_evaluate_federated_model_mocked(tmp_path):
    from fl.evaluate import evaluate_federated_model

    mock_model = MagicMock()
    mock_model.evaluate.return_value = {
        "loss": 0.1,
        "accuracy": 0.95,
        "precision_macro": 0.94,
        "recall_macro": 0.93,
        "f1_macro": 0.935,
    }
    mock_model.predict.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    mock_model.count_params.return_value = 1000
    mock_model.trainable_weights = []

    mock_dataset = MagicMock()
    mock_dataset.y = np.array([0, 1])
    mock_dataset.__len__.return_value = 2
    mock_dataset._window_at.return_value = (np.zeros((10, 19)), np.zeros((10, 5)))
    mock_dataset.to_tf_dataset.return_value = None

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    model_file = ckpt_dir / "fl_hierarchical_transformer_gru_best.keras"
    model_file.write_bytes(b"dummy")

    with patch("fl.evaluate.Path") as mock_path:
        # Route checkpoints to tmp_path
        def fake_path(p):
            if "fl_hierarchical_transformer_gru_best.keras" in str(p):
                return model_file
            return tmp_path / p
        mock_path.side_effect = fake_path

        with patch("fl.evaluate.get_model_and_dataset", return_value=(mock_model, mock_dataset)):
            with patch("fl.evaluate.validate_experiment", return_value={"num_classes": 2, "class_names": ["Normal", "Attack"]}):
                with patch("fl.evaluate.init_dagshub"):
                    with patch("fl.evaluate.mlflow"):
                        with patch("fl.evaluate.save_confusion_matrix_figure"):
                            params = {
                                "dataset": {"processed_dir": str(tmp_path)},
                                "split": {"window_size": 10},
                                "model": {},
                                "federated": {},
                                "mlflow": {},
                            }
                            evaluate_federated_model("hierarchical_transformer_gru", params)
                            mock_model.evaluate.assert_called_once()


