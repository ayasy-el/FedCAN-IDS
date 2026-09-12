"""Session-aware sliding-window dataset for the temporal MLP."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf

from data.mlp_dataset import _hex


NUMERIC_FEATURE_COLUMNS = ["DLC", *[f"Data_{i}" for i in range(8)], "Delta_Id", "Deltatime"]


def _frame_features(df: pl.DataFrame, can_id_bits: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([_hex(value) for value in df["Arbitration_ID"].to_list()], dtype=np.int32)
    bits = ((ids[:, None] >> np.arange(can_id_bits - 1, -1, -1)) & 1).astype(np.float32)
    numeric = np.column_stack([
        df["DLC"].to_numpy(),
        *[np.asarray([_hex(value) for value in df[f"Data_{i}"].to_list()]) for i in range(8)],
        df["Delta_Id"].to_numpy(),
        df["Deltatime"].to_numpy(),
    ]).astype(np.float32)
    return np.concatenate([bits, numeric], axis=1), df["Class"].to_numpy().astype(np.int32)


class MLPWindowDataset:
    """Create windows inside each session and label them with the final frame."""

    REQUIRED_COLUMNS = frozenset({
        "session_id", "Timestamp", "Arbitration_ID", "DLC", "Class",
        "Delta_Id", "Deltatime", *[f"Data_{i}" for i in range(8)],
    })

    def __init__(self, parquet_path, normalize_stats_path=None, fit_normalize_stats=False,
                 can_id_bits=11, batch_size=256, shuffle=False, window_size=16, stride=8):
        self.parquet_path = Path(parquet_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.window_size < 1 or self.stride < 1:
            raise ValueError("window_size and stride must be positive")

        df = pl.read_parquet(self.parquet_path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"MLP window input is missing required columns: {sorted(missing)}"
            )
        df = df.select(sorted(self.REQUIRED_COLUMNS)).sort(["session_id", "Timestamp"])
        sessions = [
            _frame_features(group, can_id_bits)
            for group in df.partition_by("session_id", maintain_order=True)
        ]
        self._normalize(sessions, normalize_stats_path, fit_normalize_stats, can_id_bits)

        windows, labels = [], []
        for features, session_labels in sessions:
            for start in range(0, len(features) - self.window_size + 1, self.stride):
                end = start + self.window_size
                windows.append(features[start:end].reshape(-1))
                labels.append(session_labels[end - 1])

        feature_dim = can_id_bits + len(NUMERIC_FEATURE_COLUMNS)
        self.x = np.asarray(windows, dtype=np.float32)
        if len(self.x):
            self.x = self.x.reshape(len(windows), self.window_size * feature_dim)
        else:
            self.x = np.empty((0, self.window_size * feature_dim), dtype=np.float32)
        self.y = np.asarray(labels, dtype=np.int32)
        self.input_dim = self.x.shape[1]
        self.num_classes = int(self.y.max()) + 1 if len(self.y) else 0

    @staticmethod
    def _normalize(sessions, path, fit, can_id_bits):
        if path is None or not sessions:
            return
        path = Path(path)
        numeric_start = can_id_bits
        all_numeric = np.concatenate([features[:, numeric_start:] for features, _ in sessions])
        if fit:
            mean = all_numeric.mean(axis=0)
            std = np.where(all_numeric.std(axis=0) < 1e-8, 1.0, all_numeric.std(axis=0))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"columns": NUMERIC_FEATURE_COLUMNS, "mean": mean.tolist(), "std": std.tolist()}, indent=2))
        else:
            stats = json.loads(path.read_text())
            if stats.get("columns") not in (None, NUMERIC_FEATURE_COLUMNS):
                raise ValueError("Normalization columns do not match the MLP window feature schema")
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
        for index, (features, labels) in enumerate(sessions):
            features = features.copy()
            features[:, numeric_start:] = (features[:, numeric_start:] - mean) / std
            sessions[index] = (features, labels)

    def to_tf_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        if self.shuffle:
            dataset = dataset.shuffle(min(len(self.y), 100_000), reshuffle_each_iteration=True)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
