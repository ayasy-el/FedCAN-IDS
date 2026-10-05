"""TensorFlow/NumPy input pipeline for Spatial-Temporal CAN Transformer (Jo & Kim, 2024).

Implements tokenization and sequence generation:
- Payload bytes (0x00 ~ 0xFF) -> Tokens 0 ~ 255
- CAN IDs (0x000 ~ 0x7FF) -> Offset +256 -> Tokens 256 ~ 2303
- Void token (DLC < 8 padding) -> Token 2304
- Mask token (target CAN ID masking in temporal window) -> Token 2305
- Total Vocabulary / Output Dimension: 2306
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import tensorflow as tf

from data.window_dataset_utils import load_window_references


CAN_ID_OFFSET = 256
VOID_TOKEN = 2304
MASK_TOKEN = 2305
VOCAB_SIZE = 2306


def parse_can_id(value: Union[str, int, float]) -> int:
    """Convert CAN ID hex string or int to decimal and offset by 256."""
    if isinstance(value, (int, np.integer)):
        cid = int(value)
    else:
        text = str(value).strip().upper().removeprefix("0X")
        cid = int(text, 16)
    if not 0 <= cid <= 0x7FF:
        raise ValueError(f"CAN ID outside 11-bit range: {value}")
    return cid + CAN_ID_OFFSET


def parse_payload_byte(value: Union[str, int, float, None]) -> int:
    """Convert payload byte hex string to decimal (0~255), or VOID_TOKEN if PAD."""
    if value is None or str(value).strip() in {"", "PAD", "None"}:
        return VOID_TOKEN
    if isinstance(value, (int, np.integer)):
        val = int(value)
    else:
        text = str(value).strip().upper().removeprefix("0X")
        if text in {"PAD", "R", "T"}:
            return VOID_TOKEN
        val = int(text, 16)
    if not 0 <= val <= 255:
        raise ValueError(f"Payload byte outside 8-bit range: {value}")
    return val


class CANSTTransformerDataset:
    """Lazy sliding-window dataset yielding temporal CAN ID windows, spatial payload, and targets."""

    def __init__(
        self,
        index_path: Union[str, Path],
        source_path: Optional[Union[str, Path]] = None,
        window_size: int = 64,
        batch_size: int = 1024,
        shuffle: bool = False,
        random_seed: int = 42,
    ):
        self.index_path = Path(index_path)
        self.window_size = int(window_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self._rng = np.random.default_rng(int(random_seed))

        self.index, self.sessions = load_window_references(
            self.index_path, source_path
        )
        self.session_ids = self.index["session_id"].to_list()
        self.starts = self.index["start_row_id"].to_numpy().astype(np.int64)
        self.window_sizes = self.index["window_size"].to_numpy().astype(np.int64)
        self.raw_labels = self.index["label"].to_numpy().astype(np.int64)

        # Precompute session arrays for fast indexing
        self._session_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._precompute_session_features()

    def _precompute_session_features(self) -> None:
        """Parse tokenized CAN IDs, payloads, and binary labels per session."""
        for session_id, session_df in self.sessions.items():
            row_ids = session_df["row_id"].to_numpy().astype(np.int64)

            # 1. CAN IDs offset by +256
            raw_cids = session_df["Arbitration_ID"].to_list()
            can_ids = np.asarray([parse_can_id(cid) for cid in raw_cids], dtype=np.int32)

            # 2. 8 payload bytes (0~255 or VOID_TOKEN)
            payload_cols = [session_df[f"Data_{i}"].to_list() for i in range(8)]
            n_rows = len(raw_cids)
            payloads = np.empty((n_rows, 8), dtype=np.int32)
            for j in range(8):
                col_data = payload_cols[j]
                for k in range(n_rows):
                    payloads[k, j] = parse_payload_byte(col_data[k])

            # 3. Binary class label: 0 for Normal, 1 for Attack
            classes = session_df["Class"].to_numpy().astype(np.int64)
            is_attack = (classes > 0).astype(np.int32)

            self._session_cache[session_id] = (row_ids, can_ids, payloads, is_attack)

    def __len__(self) -> int:
        return len(self.starts)

    def _window_at(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        session_id = self.session_ids[idx]
        start_row_id = self.starts[idx]
        w_size = self.window_sizes[idx]

        row_ids, can_ids, payloads, is_attack = self._session_cache[session_id]
        start_idx = int(np.searchsorted(row_ids, start_row_id))
        target_idx = start_idx + w_size - 1

        # Temporal sequence Tn(input) = [t_{n-w+1}, ..., t_{n-1}, <Mask>]
        temporal_input = can_ids[start_idx : start_idx + w_size].copy()
        temporal_input[-1] = MASK_TOKEN

        # Spatial sequence Sn(input) = [s0, ..., s7] from target frame n
        spatial_input = payloads[target_idx].copy()

        # Target CAN ID
        target_can_id = int(can_ids[target_idx])

        # Ground truth attack label (0=Normal, 1=Attack)
        attack_label = int(is_attack[target_idx])

        return temporal_input, spatial_input, target_can_id, attack_label

    def _batch_generator(
        self, indices: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
        if indices is None:
            indices = np.arange(len(self.starts))
        else:
            indices = np.asarray(indices, dtype=np.int64)

        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch_temp = []
            batch_spat = []
            batch_targets = []

            for idx in batch_idx:
                t_in, s_in, target_id, _ = self._window_at(int(idx))
                batch_temp.append(t_in)
                batch_spat.append(s_in)
                batch_targets.append(target_id)

            yield {
                "temporal_input": np.asarray(batch_temp, dtype=np.int32),
                "spatial_input": np.asarray(batch_spat, dtype=np.int32),
            }, np.asarray(batch_targets, dtype=np.int32)

    def to_tf_dataset(
        self, indices: Optional[np.ndarray] = None
    ) -> tf.data.Dataset:
        """Create a TensorFlow dataset yielding (features, target_can_id)."""
        signature = (
            {
                "temporal_input": tf.TensorSpec(
                    shape=(None, self.window_size), dtype=tf.int32
                ),
                "spatial_input": tf.TensorSpec(
                    shape=(None, 8), dtype=tf.int32
                ),
            },
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        return tf.data.Dataset.from_generator(
            lambda: self._batch_generator(indices),
            output_signature=signature,
        ).prefetch(tf.data.AUTOTUNE)
