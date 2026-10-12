"""High-performance, memory-efficient input pipeline for Lightweight CNN-LSTM (Singh et al., 2026).

Optimizations:
1. Low RAM footprint: stores payload as uint8 bytes (8 bytes/frame, ~12 MB total),
   unpacking to 8x8 binary bit-matrix [B, S, 8, 8, 1] on-the-fly via vectorized np.unpackbits.
2. Fast O(1) contiguous session slicing without searchsorted or string lookups.
3. Precomputed session-level causal temporal features:
   - can_id: normalized arbitration ID [0, 1]
   - time_delta: elapsed time since previous message with same arbitration ID
   - freq_id: causal rolling frequency within window W
   - bus_load: causal rolling message rate normalized to [0, 1]
   - time_st: normalized relative time progression
4. Exact sample label matching for frame t at window completion.
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


class CNNLSTMDataset:
    """Ultra-fast, low-memory sliding-window dataset yielding (image_input, features_input) and labels."""

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

        # 2. Precompute session arrays (compact in RAM)
        # Mapping session_id -> session index int
        self.session_names = sorted(self.sessions.keys())
        self.session_map = {name: i for i, name in enumerate(self.session_names)}

        # Arrays stored per session:
        self.session_bytes: list[np.ndarray] = []     # (N, 8) uint8 (~12 MB total)
        self.session_features: list[np.ndarray] = []  # (N, 5) float32 (~30 MB total)
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

            # 1. Compact payload: (n, 8) uint8
            byte_cols = []
            for j in range(8):
                col = df[f"Data_{j}"].to_list()
                arr = np.fromiter(
                    (_HEX_LOOKUP.get(x, 0) for x in col), dtype=np.uint8, count=n
                )
                byte_cols.append(arr)
            bytes_mat = np.column_stack(byte_cols) if n > 0 else np.zeros((0, 8), dtype=np.uint8)
            self.session_bytes.append(bytes_mat)

            # 2. 5 Temporal Features
            if n == 0:
                self.session_features.append(np.zeros((0, self.num_features), dtype=np.float32))
                continue

            # 2a. CAN ID normalized to [0, 1]
            aids = [parse_can_id(x) for x in df["Arbitration_ID"].to_list()]
            aid_arr = np.asarray(aids, dtype=np.int32)
            can_id_norm = (aid_arr.astype(np.float32) / float(0x7FF)).reshape(-1, 1)

            # 2b. Time Delta per CAN ID (fast Polars diff)
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

            # 2c. ID Frequency in rolling W messages
            freq_id = np.zeros(n, dtype=np.float32)
            kernel = np.ones(W, dtype=np.float32)
            uids = np.unique(aid_arr)
            for uid in uids:
                mask = (aid_arr == uid).astype(np.float32)
                conv = np.convolve(mask, kernel, mode="full")[:n]
                freq_id[aid_arr == uid] = conv[aid_arr == uid] / float(W)
            freq_id = freq_id.reshape(-1, 1)

            # 2d. Bus load (traffic density rate in rolling W messages, normalized)
            if n > W:
                prev_ts = np.concatenate([np.repeat(timestamps[0], W), timestamps[:-W]])
            else:
                prev_ts = np.repeat(timestamps[0], n)
            rolling_dt = timestamps - prev_ts[:n]
            rate = np.where(
                rolling_dt > 1e-6, float(W) / np.maximum(rolling_dt, 1e-4), 0.0
            )
            bus_load = np.clip(rate / 5000.0, 0.0, 1.0).astype(np.float32).reshape(-1, 1)

            # 2e. Timestamp progression
            t_min, t_max = timestamps[0], timestamps[-1]
            t_norm = (
                (timestamps - t_min) / max(t_max - t_min, 1e-6)
            ).astype(np.float32).reshape(-1, 1)

            feat_5 = np.hstack([can_id_norm, time_delta, freq_id, bus_load, t_norm]).astype(
                np.float32
            )
            self.session_features.append(feat_5)

    def _build_fast_window_pointers(self) -> None:
        """Precompute flat NumPy integer pointer arrays for O(1) batch slicing."""
        if self.num_windows == 0:
            self.window_session_idx = np.empty(0, dtype=np.int32)
            self.window_offsets = np.empty(0, dtype=np.int32)
            self.y = np.empty(0, dtype=np.int32)
            return

        self.window_session_idx = np.empty(self.num_windows, dtype=np.int32)
        self.window_offsets = np.empty(self.num_windows, dtype=np.int32)
        self.y = np.empty(self.num_windows, dtype=np.int32)

        # Vectorized population per session
        indexed_df = self.index.with_row_index("row_idx")
        for session_id_tuple, sub in indexed_df.partition_by("session_id", as_dict=True).items():
            s_name = session_id_tuple[0]
            s_idx = self.session_map[s_name]
            r0 = self.session_r0[s_idx]
            sub_row_idxs = sub["row_idx"].to_numpy()
            starts = sub["start_row_id"].to_numpy()
            sizes = sub["window_size"].to_numpy()

            offsets = starts - r0
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

        bytes_slice = self.session_bytes[s_idx][off : off + sz]
        feat_slice = self.session_features[s_idx][off : off + sz]
        img_slice = (
            np.unpackbits(bytes_slice, axis=-1)
            .reshape(sz, 8, 8, 1)
            .astype(np.float32)
        )
        return img_slice, feat_slice

    def _batch_generator(
        self, indices: np.ndarray | None = None
    ) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
        """Ultra-fast batch generator using vectorized unpackbits."""
        if indices is None:
            indices = np.arange(self.num_windows, dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        if self.shuffle:
            self._rng.shuffle(indices)

        sz = self.window_size
        b_size = self.batch_size
        n_feat = self.num_features

        for start in range(0, len(indices), b_size):
            batch_idx = indices[start : start + b_size]
            current_b = len(batch_idx)

            # Preallocate batch buffers
            batch_bytes = np.empty((current_b, sz, 8), dtype=np.uint8)
            batch_features = np.empty((current_b, sz, n_feat), dtype=np.float32)

            # Direct array slice copy
            for i, idx in enumerate(batch_idx):
                s_idx = self.window_session_idx[idx]
                off = self.window_offsets[idx]
                batch_bytes[i] = self.session_bytes[s_idx][off : off + sz]
                batch_features[i] = self.session_features[s_idx][off : off + sz]

            # Vectorized unpackbits for the entire batch in C
            batch_images = (
                np.unpackbits(batch_bytes, axis=-1)
                .reshape(current_b, sz, 8, 8, 1)
                .astype(np.float32)
            )
            batch_labels = self.y[batch_idx]

            yield {
                "image_input": batch_images,
                "features_input": batch_features,
            }, batch_labels

    def to_tf_dataset(
        self, indices: np.ndarray | None = None
    ) -> tf.data.Dataset:
        """Convert window generator to a prefetching tf.data.Dataset."""
        output_signature = (
            {
                "image_input": tf.TensorSpec(
                    shape=(None, self.window_size, 8, 8, 1), dtype=tf.float32
                ),
                "features_input": tf.TensorSpec(
                    shape=(None, self.window_size, self.num_features),
                    dtype=tf.float32,
                ),
            },
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )

        return (
            tf.data.Dataset.from_generator(
                lambda: self._batch_generator(indices),
                output_signature=output_signature,
            )
            .prefetch(tf.data.AUTOTUNE)
        )
