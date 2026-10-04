"""Streaming adapter for the Car Hacking: Attack & Defense Challenge format."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


ATTACK_COLUMNS = ("SubClass", "Class")


def _iter_chunks(path: Path):
    lf = pl.scan_csv(path, infer_schema_length=0)
    schema_cols = set(lf.collect_schema().names())
    source_label = next(
        (column for column in ATTACK_COLUMNS if column in schema_cols),
        None,
    )
    if source_label is None:
        attack_expr = pl.lit("Normal").alias("attack_type")
    else:
        attack_expr = (
            pl.col(source_label)
            .fill_null("Normal")
            .str.strip_chars()
            .alias("attack_type")
        )

    can_id_expr = (
        pl.col("Arbitration_ID")
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace("^0X", "")
        .str.strip_chars_start("0")
        .str.zfill(3)
        .alias("Arbitration_ID")
    )

    transformed = (
        lf.with_columns(
            pl.lit(path.stem).alias("session_id"),
            pl.col("Timestamp").cast(pl.Float64),
            can_id_expr,
            pl.col("DLC").cast(pl.UInt8),
            attack_expr,
        )
        .with_columns(
            pl.col("Data")
            .str.split(" ")
            .list.to_struct(fields=[f"Data_{i}" for i in range(8)])
            .alias("Bytes")
        )
        .unnest("Bytes")
        .with_columns(*[pl.col(f"Data_{i}").fill_null("PAD") for i in range(8)])
        .select(
            "session_id",
            "Timestamp",
            "Arbitration_ID",
            "DLC",
            *[f"Data_{i}" for i in range(8)],
            "attack_type",
        )
    )
    yield from transformed.collect_batches()


def read(files: list[Path]) -> pl.DataFrame:
    """Compatibility reader for small tests; production uses ``write``."""
    frames = [chunk for path in files for chunk in _iter_chunks(path)]
    if not frames:
        raise ValueError("Car Hacking Attack & Defense adapter received no valid frames")
    return pl.concat(frames, how="vertical")


def write(files: list[Path], output: Path) -> int:
    """Stream all selected Challenge files into one canonical Parquet file."""
    if not files:
        raise ValueError("Car Hacking Attack & Defense adapter received no files")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    writer = None
    total_rows = 0
    try:
        for path in files:
            file_rows = 0
            for chunk in _iter_chunks(path):
                chunk = chunk.insert_column(
                    0,
                    pl.Series(
                        "row_id",
                        range(total_rows, total_rows + len(chunk)),
                        dtype=pl.UInt64,
                    ),
                )
                table = chunk.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                total_rows += len(chunk)
                file_rows += len(chunk)
            print(f"Ingested Challenge {path.name}: {file_rows:,} frames")
        if writer is None:
            raise ValueError("Car Hacking Attack & Defense adapter found no valid frames")
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(temporary_path, output)
    return total_rows
