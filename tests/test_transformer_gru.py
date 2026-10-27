"""Unit tests for Lightweight Transformer-GRU with Nibble Lookup Table."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model.transformer_gru import (
    NibbleTokenizer,
    PayloadSliceLayer,
    build_transformer_gru,
    parameter_counts,
)


def test_nibble_tokenizer_accuracy():
    """Verify that NibbleTokenizer correctly splits 8 bytes into 16 4-bit nibble integers."""
    matrix = np.zeros((8, 8), dtype=np.float32)
    matrix[0] = [0, 0, 1, 0,  1, 0, 1, 0]
    matrix[1] = [1, 1, 1, 1,  0, 0, 0, 0]
    matrix[2] = [0, 0, 0, 0,  0, 1, 1, 1]
    matrix[3] = [1, 1, 1, 1,  1, 1, 1, 1]

    dummy = matrix[np.newaxis, ..., np.newaxis]
    tok = NibbleTokenizer()
    nibbles = tok(dummy).numpy()[0]

    assert len(nibbles) == 16
    assert nibbles[0] == 2
    assert nibbles[1] == 10
    assert nibbles[2] == 15
    assert nibbles[3] == 0
    assert nibbles[4] == 0
    assert nibbles[5] == 7
    assert nibbles[6] == 15
    assert nibbles[7] == 15


def test_payload_slice_layer():
    """Verify that PayloadSliceLayer extracts the 16 payload nibbles."""
    slice_layer = PayloadSliceLayer()
    dummy = tf.constant([[1, 2, 0, 2, 10, 15, 0, 0, 7, 15, 15, 0, 0, 5, 5, 10, 10, 3, 12]], dtype=tf.int32)
    out = slice_layer(dummy)
    assert out.shape == (1, 16)
    np.testing.assert_array_equal(out.numpy(), dummy[:, 3:].numpy())


def test_transformer_gru_architecture_and_parameter_counts():
    """Verify architecture parameter counts for default 6-class ROAD configuration."""
    model = build_transformer_gru(
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
    assert counts["total_parameters"] == 30998
    assert counts["trainable_parameters"] == 30998
    assert counts["non_trainable_parameters"] == 0


def test_transformer_gru_forward_pass_shapes():
    """Verify output shape and valid probability distribution."""
    window_size = 20
    num_features = 5
    num_classes = 6
    batch_size = 4

    model = build_transformer_gru(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
        d_model=16,
        num_heads=2,
        dim_feedforward=32,
        dropout=0.2,
        gru_hidden_size=32,
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

    output = model([dummy_tokens, dummy_features], training=False)
    assert output.shape == (batch_size, num_classes)

    sums = tf.reduce_sum(output, axis=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_transformer_gru_gradient_step():
    """Verify that all trainable weights receive gradients and update."""
    window_size = 10
    num_features = 5
    num_classes = 6
    batch_size = 2

    model = build_transformer_gru(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    dummy_tokens = tf.zeros(
        shape=(batch_size, window_size, 19), dtype=tf.int32
    )
    dummy_features = tf.zeros(
        shape=(batch_size, window_size, num_features), dtype=tf.float32
    )
    dummy_labels = tf.constant([0, 1], dtype=tf.int32)

    initial_weights = [w.numpy().copy() for w in model.trainable_weights]

    with tf.GradientTape() as tape:
        preds = model([dummy_tokens, dummy_features], training=True)
        loss = loss_fn(dummy_labels, preds)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    for i, (w_init, w_new) in enumerate(
        zip(initial_weights, model.trainable_weights)
    ):
        diff = np.abs(w_init - w_new.numpy())
        assert np.any(diff > 0.0), f"Weight at index {i} did not update!"


def test_transformer_gru_serialization():
    """Verify that model saves and loads cleanly with custom objects."""
    model = build_transformer_gru(
        window_size=10,
        num_features=5,
        num_classes=6,
    )

    dummy_tokens = tf.zeros(shape=(2, 10, 19), dtype=tf.int32)
    dummy_features = tf.zeros(shape=(2, 10, 5), dtype=tf.float32)

    pred1 = model([dummy_tokens, dummy_features], training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_model.keras"
        model.save(save_path)

        loaded_model = keras.models.load_model(
            save_path,
            custom_objects={
                "PayloadSliceLayer": PayloadSliceLayer,
                "NibbleTokenizer": NibbleTokenizer,
            },
            compile=False,
        )
        pred2 = loaded_model(
            [dummy_tokens, dummy_features], training=False
        ).numpy()

        assert np.allclose(pred1, pred2, atol=1e-5)
