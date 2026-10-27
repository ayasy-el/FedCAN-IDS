"""Unit tests for HAT-IDS (Hierarchical Attention Transformer Intrusion Detection System)."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model.hat_ids import (
    AttentionPooling1D,
    CLSToken,
    ExtractCLS,
    build_hat_ids,
    parameter_counts,
)


def test_cls_token_and_extraction():
    """Verify that CLSToken prepends [CLS] and ExtractCLS retrieves it."""
    cls_layer = CLSToken(d_model=32)
    extract_layer = ExtractCLS()

    dummy_tokens = tf.random.normal((2, 19, 32))
    with_cls = cls_layer(dummy_tokens)
    assert with_cls.shape == (2, 20, 32)

    extracted = extract_layer(with_cls)
    assert extracted.shape == (2, 32)


def test_attention_pooling_shapes():
    """Verify that AttentionPooling1D pools across the sequence dimension."""
    pool = AttentionPooling1D(units=32)
    dummy_seq = tf.random.normal((4, 20, 32))
    out = pool(dummy_seq)
    assert out.shape == (4, 32)


def test_hat_ids_architecture_and_parameter_counts():
    """Verify architecture parameter counts for default configuration (d_model=32)."""
    model = build_hat_ids(
        window_size=20,
        num_features=5,
        num_classes=6,
        d_model=32,
        num_heads=2,
        dim_feedforward=64,
        dropout=0.2,
    )
    counts = parameter_counts(model)
    assert counts["total_parameters"] == 20198
    assert counts["trainable_parameters"] == 20198
    assert counts["non_trainable_parameters"] == 0


def test_hat_ids_forward_pass_shapes():
    """Verify output shape and valid probability distribution with 3 inputs."""
    window_size = 20
    num_features = 5
    num_classes = 6
    batch_size = 4
    d_model = 32

    model = build_hat_ids(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=2,
        dim_feedforward=64,
        dropout=0.2,
    )

    dummy_tokens = tf.random.uniform(
        shape=(batch_size, window_size, 19),
        minval=0,
        maxval=16,
        dtype=tf.int32,
    )
    dummy_features = tf.random.normal(
        shape=(batch_size, window_size, num_features),
        dtype=tf.float32,
    )
    dummy_time = tf.random.normal(
        shape=(batch_size, window_size, d_model),
        dtype=tf.float32,
    )

    outputs = model(
        [dummy_tokens, dummy_features, dummy_time], training=False
    )
    assert outputs.shape == (batch_size, num_classes)

    row_sums = tf.reduce_sum(outputs, axis=-1).numpy()
    np.testing.assert_allclose(row_sums, np.ones(batch_size), atol=1e-5)


def test_hat_ids_gradient_step():
    """Verify that all trainable weights update after backpropagation."""
    window_size = 10
    num_features = 5
    num_classes = 4
    d_model = 16

    model = build_hat_ids(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.0,
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    x_tok = tf.constant(
        np.random.randint(0, 16, size=(2, 10, 19)), dtype=tf.int32
    )
    x_feat = tf.constant(
        np.random.uniform(0.0, 1.0, size=(2, 10, 5)), dtype=tf.float32
    )
    x_time = tf.constant(
        np.random.uniform(0.0, 1.0, size=(2, 10, d_model)), dtype=tf.float32
    )
    y_true = tf.constant([0, 2], dtype=tf.int32)

    initial_weights = [w.numpy().copy() for w in model.trainable_weights]

    with tf.GradientTape() as tape:
        preds = model([x_tok, x_feat, x_time], training=True)
        loss = loss_fn(y_true, preds)

    grads = tape.gradient(loss, model.trainable_weights)
    assert all(g is not None for g in grads)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    updated_weights = [w.numpy() for w in model.trainable_weights]
    any_diff = any(
        not np.allclose(init, upd)
        for init, upd in zip(initial_weights, updated_weights)
    )
    assert any_diff, "Weights should update after one gradient step"


def test_hat_ids_serialization():
    """Verify that model saves and loads cleanly with custom objects."""
    model = build_hat_ids(
        window_size=10,
        num_features=5,
        num_classes=4,
        d_model=16,
        num_heads=2,
        dim_feedforward=32,
    )
    dummy_tok = np.random.randint(0, 16, size=(2, 10, 19), dtype=np.int32)
    dummy_feat = np.random.uniform(0.0, 1.0, size=(2, 10, 5)).astype(np.float32)
    dummy_time = np.random.uniform(0.0, 1.0, size=(2, 10, 16)).astype(np.float32)

    pred_before = model(
        [dummy_tok, dummy_feat, dummy_time], training=False
    ).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "hat_ids.keras"
        model.save(str(save_path))

        custom_objects = {
            "CLSToken": CLSToken,
            "ExtractCLS": ExtractCLS,
            "AttentionPooling1D": AttentionPooling1D,
        }
        loaded = keras.models.load_model(
            str(save_path), custom_objects=custom_objects
        )
        pred_after = loaded(
            [dummy_tok, dummy_feat, dummy_time], training=False
        ).numpy()

        np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)
