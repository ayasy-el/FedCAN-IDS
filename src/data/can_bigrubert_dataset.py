"""TensorFlow input pipeline for the faithful CAN-BiGRUBERT preprocessing."""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
from transformers import AutoTokenizer


class CANBiGRUBERTDataset:
    """Load balanced window records and tokenize frames lazily by batch."""

    REQUIRED_COLUMNS = frozenset({"frames", "label"})

    def __init__(self, parquet_path, tokenizer_checkpoint, window_size, max_length,
                 batch_size=16, shuffle=False, random_seed=42):
        self.parquet_path = Path(parquet_path)
        self.window_size = int(window_size)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(int(random_seed))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True)
        frame_data = pl.read_parquet(self.parquet_path)
        if not self.REQUIRED_COLUMNS.issubset(frame_data.columns):
            raise ValueError(f"Invalid CAN-BiGRUBERT parquet schema: {frame_data.schema}")
        self.frames = frame_data["frames"].to_list()
        self.y = frame_data["label"].to_numpy().astype(np.int32)
        self.num_classes = 10
        if any(len(window) != self.window_size for window in self.frames):
            raise ValueError("All windows must have the configured window_size")

    def _batch_generator(self):
        indices = np.arange(len(self.y))
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_windows = [self.frames[int(index)] for index in batch_indices]
            flat_frames = [frame for window in batch_windows for frame in window]
            encoded = self.tokenizer(
                flat_frames,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            batch_size = len(batch_indices)
            input_ids = encoded["input_ids"].reshape(
                batch_size, self.window_size, self.max_length
            ).astype(np.int32)
            attention_mask = encoded["attention_mask"].reshape(
                batch_size, self.window_size, self.max_length
            ).astype(np.int32)
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }, self.y[batch_indices]

    def to_tf_dataset(self):
        signature = (
            {
                "input_ids": tf.TensorSpec(
                    shape=(None, self.window_size, self.max_length), dtype=tf.int32
                ),
                "attention_mask": tf.TensorSpec(
                    shape=(None, self.window_size, self.max_length), dtype=tf.int32
                ),
            },
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        return tf.data.Dataset.from_generator(
            self._batch_generator,
            output_signature=signature,
        ).prefetch(tf.data.AUTOTUNE)
