"""Input pipeline for TCT-IDS sliding-window dataset.

Reproduces packet-level 12-dimensional representations (4 nibbles CAN ID + 8 bytes payload)
and sequence-level Time-Positional Encoding (TSE) from Gong et al. (2026).
"""

from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf

from data.window_dataset_utils import load_window_references


def compute_tct_tse(
    timestamps: np.ndarray,
    alpha: float = 1e-7,
    d_model: int = 10,
    p_positions: np.ndarray | None = None,
    div_term: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Time-Positional Encoding (TSE) for a single sequence (Eq. 4, 9, 10).

    Eq. 4: tse_i = 0 if i == 0 else log2(ts_rel_i / alpha + 1)
    Eq. 9: TSE(p, 2i) = sin((p + tse_p) / 10000^(2i/d))
    Eq. 10: TSE(p, 2i+1) = cos((p + tse_p) / 10000^(2i/d))
    """
    n = len(timestamps)
    if p_positions is None:
        p_positions = np.arange(n, dtype=np.float32)
    if div_term is None:
        div_term = np.power(
            10000.0, np.arange(0, d_model, 2, dtype=np.float32) / d_model
        )

    ts_rel = timestamps - timestamps[0]
    tse_smooth = np.zeros(n, dtype=np.float32)
    if n > 1:
        tse_smooth[1:] = np.log2(
            np.maximum(ts_rel[1:] / float(alpha), 0.0) + 1.0
        )

    pos_time = p_positions + tse_smooth
    angles = pos_time[:, None] / div_term[None, :]

    tse = np.zeros((n, d_model), dtype=np.float32)
    tse[:, 0::2] = np.sin(angles)
    tse[:, 1::2] = np.cos(angles)
    return tse


def compute_batch_tct_tse(
    batch_ts: np.ndarray,
    alpha: float = 1e-7,
    d_model: int = 10,
    p_positions: np.ndarray | None = None,
    div_term: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized computation of TSE for a batch of timestamp sequences."""
    b, n = batch_ts.shape
    if p_positions is None:
        p_positions = np.arange(n, dtype=np.float32)
    if div_term is None:
        div_term = np.power(
            10000.0, np.arange(0, d_model, 2, dtype=np.float32) / d_model
        )

    ts_rel = batch_ts - batch_ts[:, :1]
    tse_smooth = np.zeros((b, n), dtype=np.float32)
    if n > 1:
        tse_smooth[:, 1:] = np.log2(
            np.maximum(ts_rel[:, 1:] / float(alpha), 0.0) + 1.0
        )

    pos_time = p_positions[None, :] + tse_smooth
    angles = pos_time[:, :, None] / div_term[None, None, :]

    tse = np.zeros((b, n, d_model), dtype=np.float32)
    tse[:, :, 0::2] = np.sin(angles)
    tse[:, :, 1::2] = np.cos(angles)
    return tse


class TCTIDSDataset:
    """Lazy sliding-window dataset yielding (message_features, time_embeddings)."""

    def __init__(
        self,
        parquet_path: str | Path,
        window_size: int = 15,
        d_model: int = 10,
        alpha: float = 1e-7,
        batch_size: int = 32,
        shuffle: bool = False,
        random_seed: int = 42,
        source_path: str | Path | None = None,
        **_ignored,
    ):
        self.parquet_path = Path(parquet_path)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(int(random_seed))

        self.index, self.sessions = load_window_references(
            self.parquet_path, source_path
        )
        self.session_ids = self.index["session_id"].to_list()
        self.starts = self.index["start_row_id"].to_numpy()
        self.window_sizes = self.index["window_size"].to_numpy()

        if len(self.window_sizes) and not np.all(
            self.window_sizes == self.window_size
        ):
            raise ValueError(
                f"All windows in dataset must match configured window_size {self.window_size}"
            )

        self.y = self.index["label"].to_numpy().astype(np.int32)

        # Precompute constants for TSE computation
        self._p_positions = np.arange(self.window_size, dtype=np.float32)
        self._div_term = np.power(
            10000.0,
            np.arange(0, self.d_model, 2, dtype=np.float32) / self.d_model,
        )

        self._session_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._precompute_session_features()

    def _precompute_session_features(self) -> None:
        """Parse frames and timestamps per session once for fast slicing."""
        for session_id, session_df in self.sessions.items():
            row_ids = session_df["row_id"].to_numpy()

            # 1. CAN ID: 4 hex nibbles
            aids = [
                str(x).upper().removeprefix("0X").zfill(4)[-4:]
                for x in session_df["Arbitration_ID"].to_list()
            ]
            nibbles = np.asarray(
                [[int(c, 16) for c in aid] for aid in aids], dtype=np.float32
            )

            # 2. Payload: 8 bytes
            payload_cols = [
                session_df[f"Data_{i}"].to_list() for i in range(8)
            ]
            payload_data = np.asarray(
                [
                    [
                        int(
                            "00"
                            if payload_cols[j][k] in (None, "PAD")
                            else payload_cols[j][k],
                            16,
                        )
                        for j in range(8)
                    ]
                    for k in range(len(aids))
                ],
                dtype=np.float32,
            )

            message_features = np.hstack([nibbles, payload_data])
            timestamps = session_df["Timestamp"].to_numpy().astype(np.float64)

            self._session_cache[session_id] = (
                row_ids,
                message_features,
                timestamps,
            )

    def _window_at(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        session_id = self.session_ids[index]
        start_row = int(self.starts[index])
        size = int(self.window_sizes[index])

        row_ids, message_features, timestamps = self._session_cache[session_id]
        offset = int(np.searchsorted(row_ids, start_row))
        if offset >= len(row_ids) or int(row_ids[offset]) != start_row:
            raise ValueError(
                f"Window start row_id {start_row} not found in session {session_id}"
            )

        window_messages = message_features[offset : offset + size]
        window_timestamps = timestamps[offset : offset + size]
        if len(window_messages) != size:
            raise ValueError(
                f"Window at index {index} has {len(window_messages)} rows, expected {size}"
            )
        return window_messages, window_timestamps

    def _batch_generator(self, indices: np.ndarray | None = None):
        if indices is None:
            indices = np.arange(len(self.y))
        else:
            indices = np.asarray(indices, dtype=np.int64)

        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            messages_list = []
            ts_list = []
            for idx in batch_idx:
                msg, ts = self._window_at(int(idx))
                messages_list.append(msg)
                ts_list.append(ts)

            batch_messages = np.asarray(messages_list, dtype=np.float32)
            batch_ts = np.asarray(ts_list, dtype=np.float64)
            batch_tse = compute_batch_tct_tse(
                batch_ts,
                alpha=self.alpha,
                d_model=self.d_model,
                p_positions=self._p_positions,
                div_term=self._div_term,
            )
            batch_labels = self.y[batch_idx]

            yield {
                "message_features": batch_messages,
                "time_embeddings": batch_tse,
            }, batch_labels

    def to_tf_dataset(
        self, indices: np.ndarray | None = None
    ) -> tf.data.Dataset:
        """Convert window generator to a prefetching tf.data.Dataset."""
        output_signature = (
            {
                "message_features": tf.TensorSpec(
                    shape=(None, self.window_size, 12), dtype=tf.float32
                ),
                "time_embeddings": tf.TensorSpec(
                    shape=(None, self.window_size, self.d_model),
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
            .repeat(1)
            .prefetch(tf.data.AUTOTUNE)
        )

    def __len__(self) -> int:
        return len(self.y)
