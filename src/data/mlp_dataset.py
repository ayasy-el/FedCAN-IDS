"""Frame-level dataset used by the feed-forward MLP baseline."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf



def _hex(value, pad=0):
    if value is None or value == "PAD":
        return pad
    return int(value, 16) if isinstance(value, str) else int(value)


def _features(df: pl.DataFrame, can_id_bits: int = 11) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([_hex(value) for value in df["Arbitration_ID"].to_list()], dtype=np.int32)
    bits = ((ids[:, None] >> np.arange(can_id_bits - 1, -1, -1)) & 1).astype(np.float32)
    numeric = np.column_stack([
        df["DLC"].to_numpy(),
        *[np.asarray([_hex(value) for value in df[f"Data_{i}"].to_list()]) for i in range(8)],
        df["Delta_Id"].to_numpy(),
        df["Deltatime"].to_numpy(),
    ]).astype(np.float32)
    return np.concatenate([bits, numeric], axis=1), df["Class"].to_numpy().astype(np.int32)


class MLPCANDataset:
    def __init__(self, parquet_path, normalize_stats_path=None, fit_normalize_stats=False,
                 can_id_bits=11, batch_size=256, shuffle=False):
        self.parquet_path = Path(parquet_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x, self.y = _features(pl.read_parquet(self.parquet_path), can_id_bits)
        self._normalize(normalize_stats_path, fit_normalize_stats)
        self.input_dim = self.x.shape[1]
        self.num_classes = int(self.y.max()) + 1 if len(self.y) else 0

    def _normalize(self, path, fit):
        if path is None:
            return
        path = Path(path)
        # Only normalize the numeric part; bit features remain in [0, 1].
        numeric_start = self.x.shape[1] - 11
        if fit:
            mean = self.x[:, numeric_start:].mean(axis=0)
            std = self.x[:, numeric_start:].std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2))
        else:
            stats = json.loads(path.read_text())
            mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
        self.x[:, numeric_start:] = (self.x[:, numeric_start:] - mean) / std

    def to_tf_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        if self.shuffle:
            dataset = dataset.shuffle(min(len(self.y), 100_000), reshuffle_each_iteration=True)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
