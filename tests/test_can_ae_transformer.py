"""Unit tests for CAN-AE-Transformer reproduction."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from model.can_ae_transformer import build_can_ae_transformer, parameter_counts
from data.can_ae_transformer_dataset import build_sinusoidal_table


def test_build_sinusoidal_table_shape():
    pe = build_sinusoidal_table(embed_dim=10, max_positions=100)
    assert pe.shape == (100, 10)
    assert pe.dtype == np.float32
    # At pos 0, sin(0) = 0, cos(0) = 1
    assert np.allclose(pe[0, 0::2], 0.0)
    assert np.allclose(pe[0, 1::2], 1.0)


def test_can_ae_transformer_forward():
    window_size = 15
    d_model = 10
    num_classes = 7
    batch_size = 4

    model = build_can_ae_transformer(
        window_size=window_size,
        d_model=d_model,
        num_heads=10,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_classes=num_classes,
    )

    dummy_messages = tf.random.uniform(
        (batch_size, window_size, 12), minval=0.0, maxval=255.0, dtype=tf.float32
    )
    dummy_tse = tf.random.uniform(
        (batch_size, window_size, d_model), minval=-1.0, maxval=1.0, dtype=tf.float32
    )

    output = model(
        {"message_features": dummy_messages, "time_embeddings": dummy_tse}
    )
    assert output.shape == (batch_size, num_classes)
    # Output probabilities must sum to 1 across classes
    sums = tf.reduce_sum(output, axis=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_parameter_counts():
    model = build_can_ae_transformer(
        window_size=15,
        d_model=10,
        num_heads=10,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_classes=7,
    )
    counts = parameter_counts(model)
    assert counts["total_parameters"] > 260_000
    assert counts["ae_parameters"] == 1114
    assert counts["classifier_parameters"] == 77
    assert counts["transformer_parameters"] == 260_988
    assert counts["trainable_parameters"] == counts["total_parameters"]
    assert counts["non_trainable_parameters"] == 0


def test_window_lookup_robustness(monkeypatch, tmp_path):
    from data.can_ae_transformer_dataset import CANAeTransformerDataset
    import polars as pl

    # Prepare dummy session and index parquet
    prepared_file = tmp_path / "prepared.parquet"
    train_file = tmp_path / "train.parquet"

    # Simulate non-monotonic or tied row_ids in a session
    sess_df = pl.DataFrame({
        "session_id": ["sess_1"] * 30,
        "Timestamp": [1.0] * 30,
        "row_id": np.arange(100, 130, dtype=np.uint64),
        "Arbitration_ID": ["153"] * 30,
        "DLC": [8] * 30,
        "Class": [0] * 30,
        **{f"Data_{i}": ["00"] * 30 for i in range(8)},
    })
    sess_df.write_parquet(prepared_file)

    index_df = pl.DataFrame({
        "window_id": ["sess_1:105"],
        "session_id": ["sess_1"],
        "start_row_id": [105],
        "window_size": [15],
        "label": [0],
    })
    index_df.write_parquet(train_file)

    ds = CANAeTransformerDataset(train_file, window_size=15, source_path=prepared_file)
    msg, tse = ds._window_at(0)
    assert msg.shape == (15, 12)
    assert tse.shape == (15, 10)


if __name__ == "__main__":
    print("Testing sinusoidal table...")
    test_build_sinusoidal_table_shape()
    print("Passed!")
    print("Testing parameter counts...")
    test_parameter_counts()
    print("Passed!")
    print("Testing forward pass...")
    test_can_ae_transformer_forward()
    print("Passed!")
    print("ALL TESTS PASSED!")
