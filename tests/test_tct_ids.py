"""Unit tests for TCT-IDS reproduction (Gong et al., 2026)."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from model.tct_ids import (
    build_tct_ids,
    enable_rdrop_training,
    parameter_counts,
)
from data.tct_ids_dataset import compute_batch_tct_tse, compute_tct_tse


def test_compute_tct_tse_shape_and_values():
    timestamps = np.array([10.0, 10.002, 10.005, 10.009, 10.015], dtype=np.float64)
    d_model = 10
    alpha = 1e-7

    tse = compute_tct_tse(timestamps, alpha=alpha, d_model=d_model)
    assert tse.shape == (5, d_model)
    assert tse.dtype == np.float32

    # At pos 0: pos = 0, tse_smooth = 0 -> sin(0) = 0, cos(0) = 1
    assert np.allclose(tse[0, 0::2], 0.0)
    assert np.allclose(tse[0, 1::2], 1.0)

    # For subsequent positions, values should be within [-1, 1]
    assert np.all(tse >= -1.0) and np.all(tse <= 1.0)


def test_compute_batch_tct_tse_equivalence():
    batch_ts = np.array(
        [
            [10.0, 10.001, 10.003],
            [20.0, 20.005, 20.010],
        ],
        dtype=np.float64,
    )
    d_model = 10
    alpha = 1e-7

    batch_tse = compute_batch_tct_tse(batch_ts, alpha=alpha, d_model=d_model)
    single_tse_0 = compute_tct_tse(batch_ts[0], alpha=alpha, d_model=d_model)
    single_tse_1 = compute_tct_tse(batch_ts[1], alpha=alpha, d_model=d_model)

    assert batch_tse.shape == (2, 3, d_model)
    assert np.allclose(batch_tse[0], single_tse_0, atol=1e-5)
    assert np.allclose(batch_tse[1], single_tse_1, atol=1e-5)


def test_tct_ids_forward_add_fusion():
    window_size = 15
    d_model = 10
    num_classes = 5
    batch_size = 4

    model = build_tct_ids(
        window_size=window_size,
        d_model=d_model,
        num_heads=1,
        num_layers=3,
        dim_feedforward=40,
        mlp_hidden_dim=40,
        tcn_filters=10,
        tcn_kernel_size=2,
        tcn_dilations=[1, 2, 4],
        dropout=0.1,
        num_classes=num_classes,
        fusion="add",
        pooling="sum",
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
    sums = tf.reduce_sum(output, axis=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_tct_ids_forward_concat_fusion():
    window_size = 15
    d_model = 10
    num_classes = 7
    batch_size = 2

    model = build_tct_ids(
        window_size=window_size,
        d_model=d_model,
        num_heads=1,
        num_layers=2,
        dim_feedforward=32,
        mlp_hidden_dim=32,
        tcn_filters=10,
        dropout=0.0,
        num_classes=num_classes,
        fusion="concat",
        pooling="mean",
    )

    dummy_messages = tf.random.uniform(
        (batch_size, window_size, 12), dtype=tf.float32
    )
    dummy_tse = tf.random.uniform(
        (batch_size, window_size, d_model), dtype=tf.float32
    )

    output = model(
        {"message_features": dummy_messages, "time_embeddings": dummy_tse}
    )
    assert output.shape == (batch_size, num_classes)
    sums = tf.reduce_sum(output, axis=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_parameter_counts_breakdown():
    model = build_tct_ids(
        window_size=15,
        d_model=10,
        num_heads=1,
        num_layers=3,
        dim_feedforward=40,
        mlp_hidden_dim=40,
        tcn_filters=10,
        num_classes=5,
    )
    counts = parameter_counts(model)
    assert counts["total_parameters"] > 0
    assert counts["mlp_parameters"] > 0
    assert counts["tcn_parameters"] > 0
    assert counts["transformer_parameters"] > 0
    assert counts["classifier_parameters"] > 0
    assert counts["trainable_parameters"] == counts["total_parameters"]


def test_rdrop_train_step():
    model = build_tct_ids(
        window_size=15,
        d_model=10,
        num_heads=1,
        num_layers=1,
        dim_feedforward=20,
        mlp_hidden_dim=20,
        tcn_filters=10,
        dropout=0.2,
        num_classes=3,
    )
    enable_rdrop_training(model, lambda_rdrop=0.1)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    dummy_messages = tf.random.uniform((8, 15, 12), dtype=tf.float32)
    dummy_tse = tf.random.uniform((8, 15, 10), dtype=tf.float32)
    dummy_labels = tf.random.uniform((8,), maxval=3, dtype=tf.int32)

    step_result = model.train_step(
        ({"message_features": dummy_messages, "time_embeddings": dummy_tse}, dummy_labels)
    )

    assert "loss" in step_result
    assert "loss_kl" in step_result
    assert float(step_result["loss"]) > 0.0
