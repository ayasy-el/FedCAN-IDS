"""MLP dataset builder for windows already materialized by the split stage."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf

from data.mlp_dataset import _hex


NUMERIC_FEATURE_COLUMNS = ["DLC", *[f"Data_{i}" for i in range(8)], "Delta_Id", "Deltatime"]


def _window_features(row, can_id_bits):
    ids = np.asarray([_hex(value) for value in row["Arbitration_ID"]], dtype=np.int32)
    bits = ((ids[:, None] >> np.arange(can_id_bits - 1, -1, -1)) & 1).astype(np.float32)
    numeric = np.column_stack([
        np.asarray(row["DLC"]),
        *[np.asarray([_hex(value) for value in row[f"Data_{i}"]]) for i in range(8)],
        np.asarray(row["Delta_Id"]),
        np.asarray(row["Deltatime"]),
    ]).astype(np.float32)
    return np.concatenate([bits, numeric], axis=1)


class MLPWindowDataset:
    REQUIRED_COLUMNS = frozenset({
        "Arbitration_ID", "DLC", "Class", "label", "Delta_Id", "Deltatime",
        *[f"Data_{i}" for i in range(8)],
    })

    def __init__(self, parquet_path, normalize_stats_path=None, fit_normalize_stats=False,
                 can_id_bits=11, batch_size=256, shuffle=False, **_ignored):
        self.parquet_path = Path(parquet_path)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        df = pl.read_parquet(self.parquet_path)
        if "label" not in df.columns:
            raise ValueError("MLPWindowDataset requires window output with a 'label' column")
        required = self.REQUIRED_COLUMNS - {"Class"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"MLP window input is missing required columns: {sorted(missing)}")

        windows = [_window_features(row, can_id_bits) for row in df.to_dicts()]
        self._normalize(windows, normalize_stats_path, fit_normalize_stats, can_id_bits)
        self.x = np.asarray([window.reshape(-1) for window in windows], dtype=np.float32)
        feature_dim = can_id_bits + len(NUMERIC_FEATURE_COLUMNS)
        if not len(self.x):
            self.x = np.empty((0, feature_dim), dtype=np.float32)
        self.y = df["label"].to_numpy().astype(np.int32)
        self.input_dim = self.x.shape[1]
        self.num_classes = int(self.y.max()) + 1 if len(self.y) else 0

    @staticmethod
    def _normalize(windows, path, fit, can_id_bits):
        if path is None or not windows:
            return
        path = Path(path)
        numeric_start = can_id_bits
        all_numeric = np.concatenate([window[:, numeric_start:] for window in windows])
        if fit:
            mean = all_numeric.mean(axis=0)
            std = np.where(all_numeric.std(axis=0) < 1e-8, 1.0, all_numeric.std(axis=0))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"columns": NUMERIC_FEATURE_COLUMNS, "mean": mean.tolist(), "std": std.tolist()}, indent=2))
        else:
            stats = json.loads(path.read_text())
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
        for index, window in enumerate(windows):
            window = window.copy()
            window[:, numeric_start:] = (window[:, numeric_start:] - mean) / std
            windows[index] = window

    def to_tf_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        if self.shuffle:
            dataset = dataset.shuffle(min(len(self.y), 100_000), reshuffle_each_iteration=True)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
