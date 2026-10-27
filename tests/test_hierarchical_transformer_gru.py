"""Unit tests for Hierarchical Spatio-Temporal Transformer-GRU model with Nibble Lookup Table."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model.hierarchical_transformer_gru import (
    AttentionPooling1D,
    CLSToken,
    ExtractCLS,
    GatedFusion,
    NibbleFrameTokenizer,
    build_hierarchical_transformer_gru,
    parameter_counts,
)


def test_nibble_frame_tokenizer_shapes():
    """Verify that NibbleFrameTokenizer embeds 19 nibbles and prepends [CLS] into 20 tokens."""
    tok = NibbleFrameTokenizer(d_model=16)
    # 19 nibbles per frame: 3 CAN ID nibbles + 16 payload nibbles (all in 0..15)
    dummy = tf.constant(
        [[1, 2, 0, 2, 10, 15, 0, 0, 7, 15, 15, 0, 0, 5, 5, 10, 10, 3, 12]],
        dtype=tf.int32,
    )
    out = tok(dummy)
    assert out.shape == (1, 20, 16)


def test_cls_extraction():
    """Verify that ExtractCLS extracts the first token vector."""
    extract = ExtractCLS()
    dummy = tf.random.normal((2, 20, 16))
    out = extract(dummy)
    assert out.shape == (2, 16)
    np.testing.assert_allclose(out.numpy(), dummy[:, 0, :].numpy())


def test_gated_fusion_shapes():
    """Verify that GatedFusion outputs correct shape."""
    fusion = GatedFusion(output_dim=16)
    spatial = tf.random.normal((4, 20, 16))
    features = tf.random.normal((4, 20, 5))
    out = fusion([spatial, features])
    assert out.shape == (4, 20, 16)


def test_hierarchical_transformer_gru_architecture_and_parameter_counts():
    """Verify architecture parameter counts for default configuration."""
    model = build_hierarchical_transformer_gru(
        window_size=20,
        num_features=5,
        num_classes=6,
        d_model=16,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.2,
        gru_hidden_size=32,
    )
    counts = parameter_counts(model)
    assert counts["total_parameters"] == 13910
    assert counts["trainable_parameters"] == 13910
    assert counts["non_trainable_parameters"] == 0


def test_hierarchical_transformer_gru_forward_pass_shapes():
    """Verify output shape and valid probability distribution with 19 nibbles input."""
    window_size = 20
    num_features = 5
    num_classes = 6
    batch_size = 4

    model = build_hierarchical_transformer_gru(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
        d_model=16,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.2,
        gru_hidden_size=32,
    )

    # Nibbles: values in 0..15
    dummy_tokens = np.random.randint(
        0, 16, size=(batch_size, window_size, 19), dtype=np.int32
    )
    dummy_features = np.random.uniform(
        0.0, 1.0, size=(batch_size, window_size, num_features)
    ).astype(np.float32)

    outputs = model([dummy_tokens, dummy_features], training=False)
    assert outputs.shape == (batch_size, num_classes)

    row_sums = tf.reduce_sum(outputs, axis=-1).numpy()
    np.testing.assert_allclose(row_sums, np.ones(batch_size), atol=1e-5)


def test_hierarchical_transformer_gru_gradient_step():
    """Verify end-to-end backpropagation and weight update with nibble tokens."""
    model = build_hierarchical_transformer_gru(
        window_size=10,
        num_features=5,
        num_classes=3,
        d_model=16,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.0,
        gru_hidden_size=16,
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    x_tok = tf.constant(
        np.random.randint(0, 16, size=(2, 10, 19)), dtype=tf.int32
    )
    x_feat = tf.constant(
        np.random.uniform(0.0, 1.0, size=(2, 10, 5)), dtype=tf.float32
    )
    y_true = tf.constant([0, 2], dtype=tf.int32)

    initial_weights = [w.numpy().copy() for w in model.trainable_weights]

    with tf.GradientTape() as tape:
        preds = model([x_tok, x_feat], training=True)
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


def test_hierarchical_transformer_gru_serialization():
    """Verify that model saves to and loads from .keras format."""
    model = build_hierarchical_transformer_gru(
        window_size=10,
        num_features=5,
        num_classes=4,
        d_model=16,
        num_heads=2,
        dim_feedforward=32,
        gru_hidden_size=16,
    )
    dummy_tok = np.random.randint(0, 16, size=(2, 10, 19), dtype=np.int32)
    dummy_feat = np.random.uniform(0.0, 1.0, size=(2, 10, 5)).astype(np.float32)

    pred_before = model([dummy_tok, dummy_feat], training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "hierarchical_transformer_gru.keras"
        model.save(str(save_path))

        custom_objects = {
            "CLSToken": CLSToken,
            "ExtractCLS": ExtractCLS,
            "NibbleFrameTokenizer": NibbleFrameTokenizer,
            "AttentionPooling1D": AttentionPooling1D,
            "GatedFusion": GatedFusion,
        }
        loaded = keras.models.load_model(
            str(save_path), custom_objects=custom_objects
        )
        pred_after = loaded([dummy_tok, dummy_feat], training=False).numpy()

        np.testing.assert_allclose(pred_before, pred_after, atol=1e-5)
