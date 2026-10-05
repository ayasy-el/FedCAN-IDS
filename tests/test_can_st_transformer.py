"""Unit tests for Spatial-Temporal CAN Transformer (Jo & Kim, 2024)."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from model.can_st_transformer import (
    build_can_st_transformer,
    parameter_counts,
)
from data.can_st_transformer_dataset import (
    parse_can_id,
    parse_payload_byte,
    CAN_ID_OFFSET,
    VOID_TOKEN,
    MASK_TOKEN,
    VOCAB_SIZE,
)


def test_token_parsing():
    # CAN ID: 0x000 -> 256, 0x7FF -> 2303
    assert parse_can_id("000") == 256
    assert parse_can_id("0X7FF") == 2303
    assert parse_can_id("2C0") == int("2C0", 16) + 256

    # Payload: 0x00 -> 0, 0xFF -> 255, PAD -> 2304
    assert parse_payload_byte("00") == 0
    assert parse_payload_byte("FF") == 255
    assert parse_payload_byte("PAD") == VOID_TOKEN
    assert parse_payload_byte(None) == VOID_TOKEN
    assert parse_payload_byte("") == VOID_TOKEN

    assert MASK_TOKEN == 2305
    assert VOCAB_SIZE == 2306


def test_can_st_transformer_forward():
    window_size = 16
    vocab_size = 2306
    d_model = 64
    batch_size = 4

    model = build_can_st_transformer(
        window_size=window_size,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=4,
        dim_feedforward=64,
        dropout=0.0,
    )

    dummy_temp = tf.random.uniform(
        (batch_size, window_size), minval=256, maxval=2304, dtype=tf.int32
    )
    dummy_spat = tf.random.uniform(
        (batch_size, 8), minval=0, maxval=255, dtype=tf.int32
    )

    output = model({"temporal_input": dummy_temp, "spatial_input": dummy_spat})
    assert output.shape == (batch_size, vocab_size)


def test_parameter_counts():
    model = build_can_st_transformer(
        window_size=64,
        vocab_size=2306,
        d_model=64,
        num_heads=4,
        num_layers=4,
        dim_feedforward=64,
    )
    counts = parameter_counts(model)
    assert counts["total"] > 400_000
    assert counts["trainable"] == counts["total"]
    assert counts["non_trainable"] == 0


def test_model_save_and_load(tmp_path):
    model = build_can_st_transformer(
        window_size=16,
        vocab_size=2306,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
    )
    save_path = tmp_path / "model.keras"
    model.save(save_path)
    loaded = tf.keras.models.load_model(save_path, compile=False)
    assert loaded is not None
    assert loaded.output_shape == (None, 2306)
