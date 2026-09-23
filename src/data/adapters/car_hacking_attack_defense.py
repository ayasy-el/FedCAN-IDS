"""Adapter for the Car Hacking: Attack & Defense Challenge format."""

from pathlib import Path

import polars as pl


ATTACK_COLUMNS = ("SubClass", "Class")


def read(files: list[Path]) -> pl.DataFrame:
    frames = []
    for path in files:
        frame = pl.read_csv(path, infer_schema_length=1000)
        source_label = next(
            (column for column in ATTACK_COLUMNS if column in frame.columns),
            None,
        )
        if source_label is None:
            frame = frame.with_columns(pl.lit("Normal").alias("attack_type"))
        else:
            frame = frame.with_columns(
                pl.col(source_label)
                .fill_null("Normal")
                .cast(pl.String)
                .alias("attack_type")
            )
        frames.append(frame.with_columns(pl.lit(path.stem).alias("session_id")))

    result = pl.concat(frames, how="diagonal")
    return (
        result.with_columns(pl.col("DLC").cast(pl.UInt8))
        .with_columns(
            pl.col("Data")
            .str.split(" ")
            .list.to_struct(fields=[f"Data_{i}" for i in range(8)])
            .alias("Bytes")
        )
        .unnest("Bytes")
        .with_columns(*[pl.col(f"Data_{i}").fill_null("PAD") for i in range(8)])
        .drop("Class", "SubClass", "Data", strict=False)
        .select(
            "session_id",
            "Timestamp",
            "Arbitration_ID",
            "DLC",
            *[f"Data_{i}" for i in range(8)],
            "attack_type",
        )
    )
