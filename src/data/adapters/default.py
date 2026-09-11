"""Adapter preserving the original FedCAN-IDS data pipeline behavior."""

from pathlib import Path

import polars as pl


SUBCLASS_MAP = {
    "Normal": 0,
    "Flooding": 1,
    "Fuzzing": 2,
    "Spoofing": 3,
    "Replay": 4,
}


def ingest(params):
    dataset = params["dataset"]
    paths = dataset["default"]
    raw_dir = Path(dataset["raw_dir"])
    output_path = Path(paths["interim_path"])
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    frames = []
    for file_path in files:
        frame = pl.read_csv(file_path, infer_schema_length=1000).with_columns(
            pl.lit(file_path.stem).alias("session_id")
        )
        if "SubClass" not in frame.columns:
            frame = frame.with_columns(pl.lit("Normal").alias("SubClass"))
        frames.append(frame)

    df = pl.concat(frames, how="diagonal")
    df = (
        df.with_columns(pl.col("DLC").cast(pl.UInt8))
        .with_columns(
            pl.when(pl.col("SubClass").is_null() & (pl.col("Class") == "Normal"))
            .then(pl.lit("Normal"))
            .otherwise(pl.col("SubClass"))
            .alias("SubClass")
        )
        .drop("Class")
        .with_columns(
            pl.col("SubClass").replace_strict(SUBCLASS_MAP).cast(pl.UInt8).alias("Class")
        )
        .drop("SubClass")
        .with_columns(
            pl.col("Data")
            .str.split(" ")
            .list.to_struct(fields=[f"Data_{i}" for i in range(8)])
            .alias("Bytes")
        )
        .unnest("Bytes")
        .with_columns(
            *[pl.col(f"Data_{i}").fill_null("PAD") for i in range(8)]
        )
        .select(
            "session_id",
            "Timestamp",
            "Arbitration_ID",
            "DLC",
            *[f"Data_{i}" for i in range(8)],
            "Class",
        )
        .with_row_index("row_id")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    print(df.head())


def split(params):
    dataset = params["dataset"]
    paths = dataset["default"]
    input_path = Path(paths["interim_path"])
    output_dir = Path(paths["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(input_path)
    all_sessions = set(df["session_id"].unique().to_list())
    val_sessions = paths["val_sessions"]
    test_sessions = paths["test_sessions"]
    unknown = (set(val_sessions) | set(test_sessions)) - all_sessions
    if unknown:
        raise RuntimeError(f"Unknown configured sessions: {sorted(unknown)}")
    overlap = set(val_sessions) & set(test_sessions)
    if overlap:
        raise RuntimeError(f"Validation/test sessions overlap: {sorted(overlap)}")

    session_sets = {
        "train": sorted(all_sessions - set(val_sessions) - set(test_sessions)),
        "val": val_sessions,
        "test": test_sessions,
    }
    for name, sessions in session_sets.items():
        result = df.filter(pl.col("session_id").is_in(sessions)).sort(
            ["session_id", "Timestamp"]
        )
        result.write_parquet(output_dir / f"{name}.parquet", compression="zstd")


def feature(params):
    dataset = params["dataset"]
    paths = dataset["default"]
    output_dir = Path(paths["featured_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        input_path = Path(paths["processed_dir"]) / f"{split_name}.parquet"
        df = pl.read_parquet(input_path).sort(["session_id", "Timestamp"])
        df = df.with_columns(
            pl.col("Timestamp")
            .diff()
            .over("session_id")
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("Deltatime"),
            pl.col("Timestamp")
            .diff()
            .over(["session_id", "Arbitration_ID"])
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("Delta_Id"),
        )
        df.write_parquet(output_dir / f"{split_name}.parquet", compression="zstd")
