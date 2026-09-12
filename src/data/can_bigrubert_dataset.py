"""TensorFlow input pipeline for compact CAN-BiGRUBERT window references."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf
from transformers import AutoTokenizer

from data.window_dataset_utils import load_window_references, materialize_window


def _frame_string(row):
    values = [str(row["Arbitration_ID"]).upper().removeprefix("0X"), format(int(row["DLC"]), "X")]
    values.extend(str(row[f"Data_{i}"]).upper() for i in range(8))
    return " ".join(values)


class CANBiGRUBERTDataset:
    """Tokenize compact window references lazily by batch."""

    REQUIRED_COLUMNS = frozenset({"frames", "label"})

    def __init__(self, parquet_path, tokenizer_checkpoint, window_size, max_length,
                 batch_size=16, shuffle=False, random_seed=42, source_path=None):
        self.parquet_path = Path(parquet_path)
        self.window_size = int(window_size)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(int(random_seed))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, use_fast=True)
        index = pl.read_parquet(self.parquet_path)
        self.legacy_frames = None
        if self.REQUIRED_COLUMNS.issubset(index.columns):
            self.index = index
            self.sessions = None
            self.legacy_frames = index["frames"].to_list()
        else:
            self.index, self.sessions = load_window_references(parquet_path, source_path)
        self.y = self.index["label"].to_numpy().astype(np.int32)
        self.num_classes = int(self.y.max()) + 1 if len(self.y) else 0
        if self.legacy_frames is None and len(self.index) and not np.all(self.index["window_size"].to_numpy() == self.window_size):
            raise ValueError("Window index size does not match the configured window_size")

    def _frames_at(self, index):
        if self.legacy_frames is not None:
            return self.legacy_frames[index]
        row = self.index.row(index, named=True)
        window = materialize_window(row, self.sessions)
        return [_frame_string(frame) for frame in window.to_dicts()]

    def _batch_generator(self):
        indices = np.arange(len(self.y))
        if self.shuffle:
            self._rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_windows = [self._frames_at(int(index)) for index in batch_indices]
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
            self._batch_generator, output_signature=signature
        ).prefetch(tf.data.AUTOTUNE)
