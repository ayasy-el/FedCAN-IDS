"""High-performance sliding-window dataset for Transformer-GRU architectures with Nibble Lookup Table.

Generates:
- tokens_input: [B, W, 19] int32 where each frame consists of 19 nibbles (values 0..15):
    - nibbles 0..2: 3 CAN Arbitration ID nibbles (12-bit representation of 11-bit ID)
    - nibbles 3..18: 16 payload nibbles (upper & lower 4 bits for Data_0..Data_7)
- features_input: [B, W, N_feat] float32 causal engineered temporal features
- labels: [B] int32 class labels
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl
import tensorflow as tf

from data.window_dataset_utils import load_window_references

# Fast lookup table for hex bytes
_HEX_LOOKUP = {f"{i:02X}": i for i in range(256)}
_HEX_LOOKUP.update({f"{i:02x}": i for i in range(256)})
_HEX_LOOKUP.update({f"{i:X}": i for i in range(16)})
_HEX_LOOKUP.update({f"{i:x}": i for i in range(16)})
_HEX_LOOKUP.update({"PAD": 0, "R": 0, "T": 0, "": 0, "None": 0})


def parse_hex_byte(value: str | int | float | None) -> int:
    """Convert hex string or int to byte value [0, 255]."""
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value) & 0xFF
    return _HEX_LOOKUP.get(str(value).strip(), 0)


def parse_can_id(value: str | int | float) -> int:
    """Convert CAN ID string or int to decimal [0, 0x7FF]."""
    if isinstance(value, (int, np.integer)):
        return int(value) & 0x7FF
    text = str(value).strip().upper().removeprefix("0X")
    try:
        return int(text, 16) & 0x7FF
    except ValueError:
        return 0


class TransformerGRUDataset:
    """Sliding-window dataset yielding (tokens_input [B, W, 19], features_input [B, W, N_feat]) and labels."""

    def __init__(
        self,
        parquet_path: str | Path,
        window_size: int = 20,
        rolling_w: int = 50,
        num_features: int = 5,
        batch_size: int = 128,
        shuffle: bool = False,
        random_seed: int = 42,
        source_path: str | Path | None = None,
        label_rule: str = "last_frame",
        **_ignored,
    ):
        self.parquet_path = Path(parquet_path)
        self.window_size = int(window_size)
        self.rolling_w = int(rolling_w)
        self.num_features = int(num_features)
        self.tokens_per_frame = 19
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.label_rule = str(label_rule)
        allowed_label_rules = {"last_frame", "any_attack"}
        if self.label_rule not in allowed_label_rules:
            raise ValueError(
                f"Unknown label_rule {self.label_rule!r}; choose one of {sorted(allowed_label_rules)}"
            )
        self._rng = np.random.default_rng(int(random_seed))

        # 1. Load compact index and sessions
        self.index, self.sessions = load_window_references(
            self.parquet_path, source_path
        )
        self.num_windows = len(self.index)

        # 2. Precompute session arrays
        self.session_names = sorted(self.sessions.keys())
        self.session_map = {name: i for i, name in enumerate(self.session_names)}

        self.session_tokens: list[np.ndarray] = []    # (N, 19) int32 in [0..15]
        self.session_features: list[np.ndarray] = []  # (N, 5) float32
        self.session_labels: list[np.ndarray] = []    # (N,) int32
        self.session_r0: list[int] = []               # start row_id per session

        self._precompute_session_features()

        # 3. Vectorized window pointer mapping (O(1) direct slice)
        self._build_fast_window_pointers()

    def _precompute_session_features(self) -> None:
        """Vectorized parsing and causal feature calculation per session."""
        W = self.rolling_w

        for name in self.session_names:
            df = self.sessions[name]
            n = len(df)
            r0 = int(df["row_id"][0]) if n > 0 else 0
            self.session_r0.append(r0)

            # Labels
            labels = df["Class"].to_numpy().astype(np.int32)
            self.session_labels.append(labels)

            # Timestamps
            timestamps = df["Timestamp"].to_numpy().astype(np.float64)

            # 1. Parse CAN ID & Payload Bytes
            aids = [parse_can_id(x) for x in df["Arbitration_ID"].to_list()]
            aid_arr = np.asarray(aids, dtype=np.int32)

            byte_cols = []
            for j in range(8):
                col = df[f"Data_{j}"].to_list()
                arr = np.fromiter(
                    (_HEX_LOOKUP.get(x, 0) for x in col), dtype=np.uint8, count=n
                )
                byte_cols.append(arr)

            bytes_mat = np.column_stack(byte_cols) if n > 0 else np.zeros((0, 8), dtype=np.uint8)

            # 2. Tokenize to 19 Nibbles (Values strictly 0..15)
            if n > 0:
                # CAN ID: 3 nibbles (12-bit representation for 11-bit ID)
                aid_n0 = ((aid_arr >> 8) & 0x0F).reshape(-1, 1)
                aid_n1 = ((aid_arr >> 4) & 0x0F).reshape(-1, 1)
                aid_n2 = (aid_arr & 0x0F).reshape(-1, 1)
                id_nibbles = np.hstack([aid_n0, aid_n1, aid_n2]).astype(np.int32)

                # Payload: 8 bytes -> 16 nibbles (upper & lower 4 bits)
                payload_hi = ((bytes_mat >> 4) & 0x0F).astype(np.int32)
                payload_lo = (bytes_mat & 0x0F).astype(np.int32)
                payload_nibbles = np.empty((n, 16), dtype=np.int32)
                payload_nibbles[:, 0::2] = payload_hi
                payload_nibbles[:, 1::2] = payload_lo

                tokens_mat = np.hstack([id_nibbles, payload_nibbles])  # (n, 19) in [0..15]
            else:
                tokens_mat = np.zeros((0, self.tokens_per_frame), dtype=np.int32)

            self.session_tokens.append(tokens_mat)

            # 3. 5 Temporal Features
            if n == 0:
                self.session_features.append(np.zeros((0, self.num_features), dtype=np.float32))
                continue

            can_id_norm = (aid_arr.astype(np.float32) / float(0x7FF)).reshape(-1, 1)

            time_delta = (
                df.with_columns(
                    pl.col("Timestamp")
                    .diff()
                    .over("Arbitration_ID")
                    .fill_null(0.0)
                    .alias("td")
                )["td"]
                .to_numpy()
                .astype(np.float32)
                .reshape(-1, 1)
            )

            freq_id = np.zeros(n, dtype=np.float32)
            kernel = np.ones(W, dtype=np.float32)
            for cid in np.unique(aid_arr):
                mask = (aid_arr == cid).astype(np.float32)
                counts = np.convolve(mask, kernel, mode="full")[:n]
                freq_id[aid_arr == cid] = counts[aid_arr == cid]
            freq_id = (freq_id / float(W)).reshape(-1, 1)

            t_diff = np.zeros(n, dtype=np.float32)
            if n > W:
                t_diff[W:] = timestamps[W:] - timestamps[:-W]
                t_diff[:W] = timestamps[:W] - timestamps[0]
            else:
                t_diff[:] = timestamps - timestamps[0]
            rate = np.zeros(n, dtype=np.float32)
            valid = t_diff > 1e-6
            rate[valid] = float(W) / t_diff[valid]
            max_rate = float(np.max(rate)) if np.max(rate) > 0 else 1.0
            bus_load = (rate / max_rate).reshape(-1, 1)

            t_span = float(timestamps[-1] - timestamps[0]) if n > 1 else 1.0
            if t_span <= 1e-6:
                t_span = 1.0
            time_st = ((timestamps - timestamps[0]) / t_span).astype(np.float32).reshape(-1, 1)

            feat_mat = np.hstack([can_id_norm, time_delta, freq_id, bus_load, time_st])
            self.session_features.append(feat_mat.astype(np.float32))

    def _build_fast_window_pointers(self) -> None:
        """Vectorized extraction of window pointer offsets and labels."""
        self.window_session_idx = np.empty(self.num_windows, dtype=np.int32)
        self.window_offsets = np.empty(self.num_windows, dtype=np.int64)
        self.y = np.empty(self.num_windows, dtype=np.int32)

        for s_idx, s_name in enumerate(self.session_names):
            mask = self.index["session_id"] == s_name
            sub_row_idxs = np.where(mask)[0]
            if len(sub_row_idxs) == 0:
                continue

            sub = self.index.filter(mask)
            r0 = self.session_r0[s_idx]
            offsets = sub["start_row_id"].to_numpy().astype(np.int64) - r0
            sizes = sub["window_size"].to_numpy().astype(np.int64)

            self.window_session_idx[sub_row_idxs] = s_idx
            self.window_offsets[sub_row_idxs] = offsets

            if self.label_rule == "last_frame":
                target_offsets = offsets + sizes - 1
                s_labels = self.session_labels[s_idx]
                self.y[sub_row_idxs] = s_labels[target_offsets]
            elif self.label_rule == "any_attack":
                self.y[sub_row_idxs] = sub["label"].to_numpy().astype(np.int32)

    def _window_at(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Extract a single window."""
        s_idx = self.window_session_idx[index]
        off = self.window_offsets[index]
        sz = self.window_size

        tok_slice = self.session_tokens[s_idx][off : off + sz]
        feat_slice = self.session_features[s_idx][off : off + sz]
        return tok_slice, feat_slice

    def _batch_generator(
        self, indices: np.ndarray | None = None
    ) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
        """Ultra-fast batch generator without unpackbits overhead."""
        if indices is None:
            indices = np.arange(self.num_windows, dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        if self.shuffle:
            self._rng.shuffle(indices)

        sz = self.window_size
        b_size = self.batch_size
        n_feat = self.num_features
        tok_count = self.tokens_per_frame

        for start in range(0, len(indices), b_size):
            batch_idx = indices[start : start + b_size]
            current_b = len(batch_idx)

            batch_tokens = np.empty((current_b, sz, tok_count), dtype=np.int32)
            batch_features = np.empty((current_b, sz, n_feat), dtype=np.float32)

            for i, idx in enumerate(batch_idx):
                s_idx = self.window_session_idx[idx]
                off = self.window_offsets[idx]
                batch_tokens[i] = self.session_tokens[s_idx][off : off + sz]
                batch_features[i] = self.session_features[s_idx][off : off + sz]

            batch_labels = self.y[batch_idx]

            yield {
                "tokens_input": batch_tokens,
                "features_input": batch_features,
            }, batch_labels

    def to_tf_dataset(
        self, indices: np.ndarray | None = None
    ) -> tf.data.Dataset:
        """Convert window generator to a prefetching tf.data.Dataset."""
        output_signature = (
            {
                "tokens_input": tf.TensorSpec(
                    shape=(None, self.window_size, self.tokens_per_frame), dtype=tf.int32
                ),
                "features_input": tf.TensorSpec(
                    shape=(None, self.window_size, self.num_features),
                    dtype=tf.float32,
                ),
            },
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: self._batch_generator(indices),
            output_signature=output_signature,
        )
        return dataset.prefetch(tf.data.AUTOTUNE)


# Backwards compatibility alias
HierarchicalTransformerGRUDataset = TransformerGRUDataset
