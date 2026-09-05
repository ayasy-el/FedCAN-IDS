"""Chronological chunks for truncated-BPTT of the stateful frame model."""

import json
from pathlib import Path
import numpy as np
import polars as pl
import tensorflow as tf

NUMERIC_FEATURE_COLUMNS = [
    "DLC",
    *[f"Data_{i}" for i in range(8)],
    "Delta_Id",
    "Deltatime",
]


def _hex(value, pad=0):
    if value is None or value == "PAD":
        return pad
    return int(value, 16) if isinstance(value, str) else int(value)


class StreamingCANDataset:
    def __init__(
        self,
        parquet_path,
        chunk_len=256,
        batch_size=32,
        shuffle=False,
        normalize_stats_path=None,
        fit_normalize_stats=False,
    ):
        self.parquet_path, self.chunk_len, self.batch_size = (
            Path(parquet_path),
            chunk_len,
            batch_size,
        )
        self.shuffle = shuffle
        df = pl.read_parquet(parquet_path).sort(["session_id", "Timestamp"])
        ids = np.array(
            [_hex(x) for x in df["Arbitration_ID"].to_list()], dtype=np.int32
        )
        numeric = np.column_stack(
            [
                df["DLC"].to_numpy(),
                *[
                    np.array([_hex(x, 0) for x in df[f"Data_{i}"].to_list()])
                    for i in range(8)
                ],
                df["Delta_Id"].to_numpy(),
                df["Deltatime"].to_numpy(),
            ]
        ).astype(np.float32)
        self.ids, self.numeric, self.labels = (
            ids,
            numeric,
            df["Class"].to_numpy().astype(np.int32),
        )
        self.session_ids = np.array(df["session_id"].to_list())
        self._normalize(normalize_stats_path, fit_normalize_stats)
        self.starts = [
            i
            for i in range(0, len(self.labels) - chunk_len + 1, chunk_len)
            if self.session_ids[i] == self.session_ids[i + chunk_len - 1]
        ]
        self.window_labels = (
            np.concatenate([self.labels[i : i + chunk_len] for i in self.starts])
            if self.starts
            else np.array([], dtype=np.int32)
        )
        self.num_samples = len(self.starts)

    def _normalize(self, path, fit):
        if path is None:
            return
        path = Path(path)
        if fit:
            mean, std = self.numeric.mean(0), self.numeric.std(0)
            std = np.where(std < 1e-8, 1.0, std)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "columns": NUMERIC_FEATURE_COLUMNS,
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                    },
                    indent=2,
                )
            )
        else:
            stats = json.loads(path.read_text())
            if stats["columns"] != NUMERIC_FEATURE_COLUMNS:
                raise ValueError(
                    "Normalization columns do not match the streaming feature schema"
                )
            mean, std = (
                np.array(stats["mean"], np.float32),
                np.array(stats["std"], np.float32),
            )
        self.numeric = (self.numeric - mean) / std

    def generator(self):
        for start in self.starts:
            end = start + self.chunk_len
            yield (
                {
                    "can_id": self.ids[start:end],
                    "numeric": self.numeric[start:end],
                    "stream_id": self.session_ids[start].encode(),
                },
                self.labels[start:end],
            )

    def to_tf_dataset(self):
        sig = (
            {
                "can_id": tf.TensorSpec((self.chunk_len,), tf.int32),
                "numeric": tf.TensorSpec((self.chunk_len, 11), tf.float32),
                "stream_id": tf.TensorSpec((), tf.string),
            },
            tf.TensorSpec((self.chunk_len,), tf.int32),
        )
        ds = tf.data.Dataset.from_generator(self.generator, output_signature=sig)
        if self.shuffle:
            ds = ds.shuffle(min(self.num_samples, 10000), reshuffle_each_iteration=True)
        return ds.batch(self.batch_size, drop_remainder=False).prefetch(
            tf.data.AUTOTUNE
        )
