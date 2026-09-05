"""Session-parallel chronological chunks for stateful Keras training."""
import json
from pathlib import Path
import numpy as np
import polars as pl
import tensorflow as tf

NUMERIC_FEATURE_COLUMNS = ["DLC", *[f"Data_{i}" for i in range(8)], "Delta_Id", "Deltatime"]


def _hex(value, pad=0):
    if value is None or value == "PAD":
        return pad
    return int(value, 16) if isinstance(value, str) else int(value)


class StreamingCANDataset:
    def __init__(self, parquet_path, chunk_len=256, batch_size=4, shuffle=False,
                 normalize_stats_path=None, fit_normalize_stats=False):
        self.parquet_path, self.chunk_len, self.batch_size, self.shuffle = Path(parquet_path), chunk_len, batch_size, shuffle
        df = pl.read_parquet(parquet_path).sort(["session_id", "Timestamp"])
        self.sessions = []
        for session, group in df.partition_by("session_id", as_dict=True, maintain_order=True).items():
            ids = np.array([_hex(x) for x in group["Arbitration_ID"].to_list()], dtype=np.int32)
            numeric = np.column_stack([
                group["DLC"].to_numpy(),
                *[np.array([_hex(x) for x in group[f"Data_{i}"].to_list()]) for i in range(8)],
                group["Delta_Id"].to_numpy(), group["Deltatime"].to_numpy(),
            ]).astype(np.float32)
            labels = group["Class"].to_numpy().astype(np.int32)
            self.sessions.append((str(session[0]) if isinstance(session, tuple) else str(session), ids, numeric, labels))
        self._normalize(normalize_stats_path, fit_normalize_stats)
        self.groups = [self.sessions[i:i + batch_size] for i in range(0, len(self.sessions), batch_size)]
        self.num_samples = sum(max(1, max(len(s[1]) // chunk_len for s in group)) for group in self.groups)
        self.window_labels = np.concatenate([s[3] for s in self.sessions]) if self.sessions else np.array([], dtype=np.int32)

    def _normalize(self, path, fit):
        if path is None or not self.sessions:
            return
        path = Path(path)
        all_numeric = np.concatenate([s[2] for s in self.sessions])
        if fit:
            mean, std = all_numeric.mean(0), all_numeric.std(0)
            std = np.where(std < 1e-8, 1.0, std)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"columns": NUMERIC_FEATURE_COLUMNS, "mean": mean.tolist(), "std": std.tolist()}, indent=2))
        else:
            stats = json.loads(path.read_text())
            if stats["columns"] != NUMERIC_FEATURE_COLUMNS:
                raise ValueError("Normalization columns do not match the streaming feature schema")
            mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
        for i, (name, ids, numeric, labels) in enumerate(self.sessions):
            self.sessions[i] = (name, ids, (numeric - mean) / std, labels)

    def generator(self):
        for group in self.groups:
            max_chunks = max(len(s[1]) // self.chunk_len for s in group)
            for chunk_no in range(max_chunks):
                ids = np.zeros((self.batch_size, self.chunk_len), np.int32)
                numeric = np.zeros((self.batch_size, self.chunk_len, 11), np.float32)
                labels = np.zeros((self.batch_size, self.chunk_len), np.int32)
                valid = np.zeros((self.batch_size, self.chunk_len), np.float32)
                stream_ids = np.full((self.batch_size,), b"__inactive__", dtype="S64")
                for slot, (name, session_ids, session_numeric, session_labels) in enumerate(group):
                    start, end = chunk_no * self.chunk_len, (chunk_no + 1) * self.chunk_len
                    if end <= len(session_ids):
                        ids[slot], numeric[slot], labels[slot] = session_ids[start:end], session_numeric[start:end], session_labels[start:end]
                        valid[slot] = 1.0
                        stream_ids[slot] = name.encode()
                yield ({"can_id": ids, "numeric": numeric, "stream_id": stream_ids, "valid_mask": valid}, labels)

    def to_tf_dataset(self):
        sig = ({"can_id": tf.TensorSpec((self.batch_size, self.chunk_len), tf.int32), "numeric": tf.TensorSpec((self.batch_size, self.chunk_len, 11), tf.float32), "stream_id": tf.TensorSpec((self.batch_size,), tf.string), "valid_mask": tf.TensorSpec((self.batch_size, self.chunk_len), tf.float32)}, tf.TensorSpec((self.batch_size, self.chunk_len), tf.int32))
        return tf.data.Dataset.from_generator(self.generator, output_signature=sig).prefetch(tf.data.AUTOTUNE)
