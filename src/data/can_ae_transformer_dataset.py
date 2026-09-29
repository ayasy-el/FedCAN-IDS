"""Input pipeline for CAN-AE-Transformer sliding-window dataset.

Reproduces packet-level 12-dimensional representations (4 nibbles CAN ID + 8 bytes payload)
and sequence-level Time-Positional Encoding (TSE) from Le et al. (2024).
"""

from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf

from data.window_dataset_utils import load_window_references


def build_sinusoidal_table(
    embed_dim: int = 10, max_positions: int = 10000
) -> np.ndarray:
    """Precompute sinusoidal positional encodings for discrete time positions."""
    pe = np.zeros((max_positions, embed_dim), dtype=np.float32)
    position = np.arange(max_positions)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


class CANAeTransformerDataset:
    """Lazy sliding-window dataset yielding (message_features, time_embeddings)."""

    def __init__(
        self,
        parquet_path: str | Path,
        window_size: int = 15,
        d_model: int = 10,
        granularity: float = 1e-8,
        max_time_position: int = 10000,
        batch_size: int = 32,
        shuffle: bool = False,
        random_seed: int = 42,
        source_path: str | Path | None = None,
        **_ignored,
    ):
        self.parquet_path = Path(parquet_path)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.granularity = float(granularity)
        self.max_time_position = int(max_time_position)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(int(random_seed))

        self.index, self.sessions = load_window_references(
            self.parquet_path, source_path
        )
        self.session_ids = self.index["session_id"].to_list()
        self.starts = self.index["start_row_id"].to_numpy()
        self.window_sizes = self.index["window_size"].to_numpy()
        self.y = self.index["label"].to_numpy().astype(np.int32)
        self.num_classes = int(self.y.max()) + 1 if len(self.y) else 0

        if len(self.window_sizes) and not np.all(
            self.window_sizes == self.window_size
        ):
            raise ValueError(
                "All windows in dataset must match configured window_size"
            )

        self.pe_table = build_sinusoidal_table(
            self.d_model, self.max_time_position
        )
        self._session_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._precompute_session_features()

    def _precompute_session_features(self) -> None:
        """Parse frames and timestamps per session once for ultra-fast slicing."""
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
                    [int(str(payload_cols[j][k] or "00"), 16) for j in range(8)]
                    for k in range(len(aids))
                ],
                dtype=np.float32,
            )

            # Combined 12-dimensional message features
            message_features = np.hstack([nibbles, payload_data])

            # 3. Time smoothing (Eq. 9) & TSE lookup
            ts = session_df["Timestamp"].to_numpy()
            scaled_ts = np.round(ts / self.granularity) + 1.0
            # log2(scaled_ts) with smooth factor
            smoothed_idx = np.round(np.log2(np.maximum(scaled_ts, 1.0))).astype(
                np.int32
            )
            smoothed_idx = np.clip(
                smoothed_idx, 0, self.max_time_position - 1
            )
            tse_features = self.pe_table[smoothed_idx]

            self._session_cache[session_id] = (
                row_ids,
                message_features,
                tse_features,
            )

    def _window_at(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        session_id = self.session_ids[index]
        start_row = int(self.starts[index])
        size = int(self.window_sizes[index])

        row_ids, message_features, tse_features = self._session_cache[session_id]
        offset = int(np.searchsorted(row_ids, start_row))
        if offset >= len(row_ids) or int(row_ids[offset]) != start_row:
            raise ValueError(
                f"Window start row_id {start_row} not found in session {session_id}"
            )

        window_messages = message_features[offset : offset + size]
        window_tse = tse_features[offset : offset + size]
        if len(window_messages) != size:
            raise ValueError(
                f"Window at index {index} has {len(window_messages)} rows, expected {size}"
            )
        return window_messages, window_tse

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
            tse_list = []
            for idx in batch_idx:
                msg, tse = self._window_at(int(idx))
                messages_list.append(msg)
                tse_list.append(tse)

            batch_messages = np.asarray(messages_list, dtype=np.float32)
            batch_tse = np.asarray(tse_list, dtype=np.float32)
            batch_labels = self.y[batch_idx]

            yield {
                "message_features": batch_messages,
                "time_embeddings": batch_tse,
            }, batch_labels

    def to_tf_dataset(
        self, indices: np.ndarray | None = None
    ) -> tf.data.Dataset:
        signature = (
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
        return tf.data.Dataset.from_generator(
            lambda: self._batch_generator(indices),
            output_signature=signature,
        ).prefetch(tf.data.AUTOTUNE)
