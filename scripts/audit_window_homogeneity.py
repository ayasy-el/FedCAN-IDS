"""Fail-fast audit for multiple attack labels in a window."""

import argparse

import polars as pl


def audit(path):
    df = pl.read_parquet(path)
    required = {"frames", "label", "frame_labels"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Expected window parquet with frames, frame_labels, and label columns"
        )

    columns = [
        column
        for column in (
            "window_id",
            "session_id",
            "start_row_id",
            "label",
            "frame_labels",
            "frames",
        )
        if column in df.columns
    ]
    for row in df.select(columns).iter_rows(named=True):
        # Labels 0 and 5 are both benign and may coexist. Only distinct
        # attack labels are invalid.
        attack_labels = sorted({
            int(label) for label in row["frame_labels"] if int(label) not in (0, 5)
        })
        if len(attack_labels) > 1:
            print(f"Found a window containing multiple attack labels in {path}")
            print("Benign labels 0 and 5 are allowed.")
            print("\n--- multiple-attack window ---")
            print(f"attack_labels: {attack_labels}")
            for column in columns:
                print(f"{column}: {row[column]}")
            raise ValueError(f"Window contains multiple attack labels in {path}")

    print(
        f"{path}: {len(df):,} valid windows; "
        f"labels={sorted(df['label'].unique().to_list())}"
    )
    lengths = df.select(pl.col("frames").list.len().alias("length"))
    print(lengths.group_by("length").len().sort("length"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="+", help="Window parquet files to inspect")
    args = parser.parse_args()
    for path in args.path:
        audit(path)
