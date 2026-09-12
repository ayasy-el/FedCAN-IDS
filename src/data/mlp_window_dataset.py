"""Lazy MLP dataset builder for compact window references."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf

from data.mlp_dataset import _hex
from data.window_dataset_utils import load_window_references, materialize_window


NUMERIC_FEATURE_COLUMNS = ["DLC", *[f"Data_{i}" for i in range(8)], "Delta_Id", "Deltatime"]


def _frame_features(df, can_id_bits):
    ids = np.asarray([_hex(value) for value in df["Arbitration_ID"].to_list()], dtype=np.int32)
    bits = ((ids[:, None] >> np.arange(can_id_bits - 1, -1, -1)) & 1).astype(np.float32)
    numeric = np.column_stack([
        df["DLC"].to_numpy(),
        *[np.asarray([_hex(value) for value in df[f"Data_{i}"].to_list()]) for i in range(8)],
        df["Delta_Id"].to_numpy(),
        df["Deltatime"].to_numpy(),
    ]).astype(np.float32)
    return np.concatenate([bits, numeric], axis=1)


class MLPWindowDataset:
    def __init__(self, parquet_path, normalize_stats_path=None, fit_normalize_stats=False,
                 can_id_bits=11, batch_size=256, shuffle=False, source_path=None, **_ignored):
        self.parquet_path = Path(parquet_path)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.can_id_bits = int(can_id_bits)
        self.index, self.sessions = load_window_references(parquet_path, source_path)
        self.session_ids = self.index["session_id"].to_list()
        self.starts = self.index["start_row_id"].to_numpy()
        self.window_sizes = self.index["window_size"].to_numpy()
        self.y = self.index["label"].to_numpy().astype(np.int32)
        self.window_size = int(self.window_sizes[0]) if len(self.window_sizes) else 0
        if len(self.window_sizes) and not np.all(self.window_sizes == self.window_size):
            raise ValueError("All windows in one dataset must have the same window_size")
        self.feature_dim = self.can_id_bits + len(NUMERIC_FEATURE_COLUMNS)
        self.input_dim = self.window_size * self.feature_dim
        self.num_classes = int(self.y.max()) + 1 if len(self.y) else 0
        self.mean, self.std = self._load_or_fit_stats(normalize_stats_path, fit_normalize_stats)

    def _features_at(self, index):
        row = {
            "window_id": self.index["window_id"][index],
            "session_id": self.session_ids[index],
            "start_row_id": int(self.starts[index]),
            "window_size": int(self.window_sizes[index]),
            "label": int(self.y[index]),
        }
        return _frame_features(materialize_window(row, self.sessions), self.can_id_bits)

    def _load_or_fit_stats(self, path, fit):
        if path is None and not fit:
            return None, None
        if fit:
            total = np.zeros(len(NUMERIC_FEATURE_COLUMNS), dtype=np.float64)
            squared = np.zeros_like(total)
            count = 0
            for index in range(len(self.y)):
                numeric = self._features_at(index)[:, self.can_id_bits:].astype(np.float64)
                total += numeric.sum(axis=0)
                squared += np.square(numeric).sum(axis=0)
                count += len(numeric)
            mean = total / max(count, 1)
            variance = np.maximum(squared / max(count, 1) - np.square(mean), 0.0)
            std = np.where(np.sqrt(variance) < 1e-8, 1.0, np.sqrt(variance))
            if path is not None:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"columns": NUMERIC_FEATURE_COLUMNS, "mean": mean.tolist(), "std": std.tolist()}, indent=2))
            return mean.astype(np.float32), std.astype(np.float32)
        stats = json.loads(Path(path).read_text())
        if stats.get("columns") not in (None, NUMERIC_FEATURE_COLUMNS):
            raise ValueError("Normalization columns do not match the MLP window feature schema")
        return np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)

    def _batch_generator(self, indices):
        indices = np.asarray(indices, dtype=np.int64).copy()
        if self.shuffle:
            np.random.default_rng().shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = []
            for index in batch_indices:
                features = self._features_at(int(index)).copy()
                if self.mean is not None:
                    features[:, self.can_id_bits:] = (
                        features[:, self.can_id_bits:] - self.mean
                    ) / self.std
                batch.append(features.reshape(-1))
            yield np.asarray(batch, dtype=np.float32), self.y[batch_indices]

    def to_tf_dataset(self, indices=None):
        indices = np.arange(len(self.y)) if indices is None else np.asarray(indices)
        signature = (
            tf.TensorSpec(shape=(None, self.input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        return tf.data.Dataset.from_generator(
            lambda: self._batch_generator(indices), output_signature=signature
        ).prefetch(tf.data.AUTOTUNE)
