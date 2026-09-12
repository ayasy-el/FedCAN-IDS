"""Fail-fast audit for multiple attack labels in expanded or compact windows."""

import argparse

import polars as pl

from data.window_dataset_utils import load_window_references, materialize_window


def _expanded_rows(df):
    columns = [
        column
        for column in (
            "window_id", "session_id", "start_row_id", "label", "frame_labels", "frames"
        )
        if column in df.columns
    ]
    for row in df.select(columns).iter_rows(named=True):
        yield row


def _compact_rows(index_path, prepared_path):
    index, sessions = load_window_references(index_path, prepared_path)
    for row in index.iter_rows(named=True):
        window = materialize_window(row, sessions)
        yield {
            "window_id": row["window_id"],
            "session_id": row["session_id"],
            "start_row_id": row["start_row_id"],
            "label": row["label"],
            "frame_labels": window["Class"].to_list(),
            "frames": window.to_dicts(),
        }


def audit(path, prepared_path=None):
    df = pl.read_parquet(path)
    if "frame_labels" in df.columns and "frames" in df.columns:
        rows = _expanded_rows(df)
    elif {"window_id", "session_id", "start_row_id", "window_size", "label"}.issubset(df.columns):
        if prepared_path is None:
            raise ValueError("Compact windows require --prepared PATH for label auditing")
        rows = _compact_rows(path, prepared_path)
    else:
        raise ValueError("Unsupported expanded or compact window schema")

    checked = 0
    for row in rows:
        checked += 1
        attack_labels = sorted({
            int(label) for label in row["frame_labels"] if int(label) not in (0, 5)
        })
        if len(attack_labels) > 1:
            print(f"Found a window containing multiple attack labels in {path}")
            print("Benign labels 0 and 5 are allowed.")
            print("\n--- multiple-attack window ---")
            print(f"attack_labels: {attack_labels}")
            for column, value in row.items():
                print(f"{column}: {value}")
            raise ValueError(f"Window contains multiple attack labels in {path}")

    print(f"{path}: {checked:,} valid windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="+", help="Window parquet files to inspect")
    parser.add_argument(
        "--prepared",
        help="Prepared frame parquet required when auditing compact window indexes",
    )
    args = parser.parse_args()
    for path in args.path:
        audit(path, args.prepared)
