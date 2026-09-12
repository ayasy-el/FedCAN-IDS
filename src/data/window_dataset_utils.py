"""Shared helpers for reconstructing compact window references."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl


WINDOW_INDEX_COLUMNS = {"window_id", "session_id", "start_row_id", "window_size", "label"}
SOURCE_COLUMNS = {
    "session_id", "Timestamp", "row_id", "Arbitration_ID", "DLC", "Class",
    "Delta_Id", "Deltatime", *[f"Data_{i}" for i in range(8)],
}


@lru_cache(maxsize=2)
def _load_sessions(prepared_path):
    source = pl.read_parquet(prepared_path, columns=sorted(SOURCE_COLUMNS)).sort(
        ["session_id", "Timestamp"]
    )
    return {
        str(session["session_id"][0]): session
        for session in source.partition_by("session_id", maintain_order=True)
    }


def load_window_references(index_path, prepared_path):
    index = pl.read_parquet(index_path)
    if not WINDOW_INDEX_COLUMNS.issubset(index.columns):
        raise ValueError(f"Invalid compact window index schema: {index.schema}")
    prepared_path = str(Path(prepared_path).resolve())
    sessions = _load_sessions(prepared_path)
    missing = set(index["session_id"].unique().to_list()) - set(sessions)
    if missing:
        raise ValueError(f"Window index references unknown sessions: {sorted(missing)}")
    return index, sessions


def materialize_window(row, sessions):
    session = sessions[str(row["session_id"])]
    start = int(row["start_row_id"])
    size = int(row["window_size"])
    row_ids = session["row_id"].to_numpy()
    offset = int(np.searchsorted(row_ids, start))
    if offset >= len(row_ids) or int(row_ids[offset]) != start:
        raise ValueError(f"Window start row_id {start} not found in session {row['session_id']}")
    window = session.slice(offset, size)
    if len(window) != size:
        raise ValueError(f"Window {row['window_id']} has only {len(window)} rows; expected {size}")
    return window
