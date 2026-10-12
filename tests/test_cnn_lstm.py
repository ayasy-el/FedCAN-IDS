"""Unit tests for Lightweight CNN-LSTM reproduction (Singh et al., 2026)."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from model.cnn_lstm import build_cnn_lstm, parameter_counts
from data.cnn_lstm_dataset import CNNLSTMDataset, parse_can_id, parse_hex_byte
import polars as pl
import pyarrow.parquet as pq


def test_cnn_lstm_architecture_and_parameter_counts():
    # Car-Hacking 5-class configuration
    model_5class = build_cnn_lstm(
        window_size=20,
        num_features=5,
        num_classes=5,
        cnn_filters=16,
        cnn_kernel_size=3,
        dropout=0.2,
        lstm_hidden_size=32,
    )
    counts_5 = parameter_counts(model_5class)

    # Conv2D: (1 * 3 * 3 + 1) * 16 = 160
    # LSTM: 4 * (261 + 32 + 1) * 32 = 37632
    # Dense: 32 * 5 + 5 = 165
    # Total: 37957
    assert counts_5["total_parameters"] == 37957
    assert counts_5["trainable_parameters"] == 37957
    assert counts_5["non_trainable_parameters"] == 0

    # ROAD 6-class configuration (Masquerade)
    model_6class = build_cnn_lstm(
        window_size=20,
        num_features=5,
        num_classes=6,
        cnn_filters=16,
        cnn_kernel_size=3,
        dropout=0.2,
        lstm_hidden_size=32,
    )
    counts_6 = parameter_counts(model_6class)
    # Dense: 32 * 6 + 6 = 198 -> Total: 37990
    assert counts_6["total_parameters"] == 37990


def test_cnn_lstm_forward_pass_shapes():
    window_size = 20
    num_features = 5
    num_classes = 6
    batch_size = 4

    model = build_cnn_lstm(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
        cnn_filters=16,
        cnn_kernel_size=3,
        dropout=0.2,
        lstm_hidden_size=32,
    )

    dummy_image = tf.random.uniform(
        shape=(batch_size, window_size, 8, 8, 1), minval=0, maxval=2, dtype=tf.float32
    )
    dummy_features = tf.random.normal(
        shape=(batch_size, window_size, num_features), dtype=tf.float32
    )

    output = model([dummy_image, dummy_features], training=False)
    assert output.shape == (batch_size, num_classes)
    # Softmax probabilities should sum to 1.0 along class axis
    sums = tf.reduce_sum(output, axis=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_cnn_lstm_gradient_step():
    window_size = 15
    num_features = 5
    num_classes = 6
    batch_size = 2

    model = build_cnn_lstm(
        window_size=window_size,
        num_features=num_features,
        num_classes=num_classes,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    dummy_image = tf.zeros(shape=(batch_size, window_size, 8, 8, 1), dtype=tf.float32)
    dummy_features = tf.zeros(shape=(batch_size, window_size, num_features), dtype=tf.float32)
    dummy_labels = tf.constant([0, 1], dtype=tf.int32)

    initial_weights = [w.numpy() for w in model.trainable_weights]

    with tf.GradientTape() as tape:
        logits = model([dummy_image, dummy_features], training=True)
        loss = loss_fn(dummy_labels, logits)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Verify weights actually updated
    updated = False
    for initial, current in zip(initial_weights, model.trainable_weights):
        if not np.allclose(initial, current.numpy()):
            updated = True
            break
    assert updated, "Model weights did not update after gradient descent step"


def test_parsing_helpers():
    assert parse_hex_byte("00") == 0
    assert parse_hex_byte("FF") == 255
    assert parse_hex_byte("7B") == 123
    assert parse_hex_byte("PAD") == 0
    assert parse_hex_byte(None) == 0

    assert parse_can_id("000") == 0
    assert parse_can_id("7FF") == 2047
    assert parse_can_id("0354") == 852
    assert parse_can_id(100) == 100


def test_cnn_lstm_dataset_pipeline(tmp_path):
    # Create mock session data (prepared.parquet)
    n_frames = 40
    window_size = 20
    timestamps = np.linspace(0.0, 1.0, n_frames)
    prepared_df = pl.DataFrame({
        "row_id": np.arange(n_frames, dtype=np.uint64),
        "session_id": ["sess_1"] * n_frames,
        "Timestamp": timestamps,
        "Arbitration_ID": ["123" if i % 2 == 0 else "456" for i in range(n_frames)],
        "DLC": [8] * n_frames,
        "Data_0": ["01"] * n_frames,
        "Data_1": ["02"] * n_frames,
        "Data_2": ["03"] * n_frames,
        "Data_3": ["04"] * n_frames,
        "Data_4": ["05"] * n_frames,
        "Data_5": ["06"] * n_frames,
        "Data_6": ["07"] * n_frames,
        "Data_7": ["08"] * n_frames,
        "Class": [1 if i >= 30 else 0 for i in range(n_frames)],
    })
    prepared_path = tmp_path / "prepared.parquet"
    prepared_df.write_parquet(prepared_path)

    # Create mock window index (train.parquet)
    index_df = pl.DataFrame({
        "window_id": ["sess_1:0", "sess_1:15"],
        "session_id": ["sess_1", "sess_1"],
        "start_row_id": [0, 15],
        "window_size": [window_size, window_size],
        "label": [0, 1],
    })
    index_path = tmp_path / "train.parquet"
    index_df.write_parquet(index_path)

    dataset = CNNLSTMDataset(
        parquet_path=index_path,
        source_path=prepared_path,
        window_size=window_size,
        batch_size=2,
        shuffle=False,
        label_rule="last_frame",
    )

    tf_ds = dataset.to_tf_dataset()
    for batch_inputs, batch_labels in tf_ds.take(1):
        images = batch_inputs["image_input"].numpy()
        features = batch_inputs["features_input"].numpy()
        labels = batch_labels.numpy()

        assert images.shape == (2, window_size, 8, 8, 1)
        assert features.shape == (2, window_size, 5)
        assert labels.shape == (2,)

        # In first window [0..19], frame 19 has label 0
        assert labels[0] == 0
        # In second window [15..34], frame 34 has label 1 (since frames >= 30 have Class=1)
        assert labels[1] == 1

    # Test any_attack rule uses the window index label directly
    ds_any = CNNLSTMDataset(
        parquet_path=index_path,
        source_path=prepared_path,
        window_size=window_size,
        batch_size=2,
        shuffle=False,
        label_rule="any_attack",
    )
    assert np.array_equal(ds_any.y, [0, 1])

    # Test invalid label_rule raises ValueError
    import pytest
    with pytest.raises(ValueError, match="Unknown label_rule"):
        CNNLSTMDataset(
            parquet_path=index_path,
            source_path=prepared_path,
            window_size=window_size,
            label_rule="invalid_rule",
        )

