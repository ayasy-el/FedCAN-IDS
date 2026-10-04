"""Streaming adapter for CICIoV2024 hexadecimal CSV files."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


DATA_COLUMNS = [f"DATA_{i}" for i in range(8)]
REQUIRED_COLUMNS = {"ID", "label", "specific_class", *DATA_COLUMNS}


def _hex_byte(value):
    if value is None:
        return "PAD"
    text = str(value).strip().upper().removeprefix("0X")
    if not text:
        return "PAD"
    number = int(float(text)) if "." in text else int(text, 16)
    if not 0 <= number <= 255:
        raise ValueError(f"Invalid CAN byte value: {value!r}")
    return f"{number:02X}"


def _dlc(value):
    if value is None or not str(value).strip():
        return 8
    text = str(value).strip().strip("[]")
    number = int(float(text)) if "." in text else int(text)
    if not 0 <= number <= 8:
        raise ValueError(f"Invalid DLC value: {value!r}")
    return number


def _attack_type(value):
    value = str(value).strip()
    if value.upper() == "BENIGN":
        return "BENIGN"
    if value.upper() == "DOS":
        return "DoS"
    return value.upper()


def _can_id(value):
    """Validate and canonicalize an 11-bit hexadecimal CAN ID."""
    text = str(value).strip().upper().removeprefix("0X")
    if "." in text:
        number = int(float(text))
    else:
        number = int(text, 16)
    if not 0 <= number <= 0x7FF:
        raise ValueError(f"CAN ID is outside the 11-bit range: {value!r}")
    return f"{number:03X}"


def _iter_chunks(path: Path):
    lf = pl.scan_csv(path, infer_schema_length=0)
    schema_cols = set(lf.collect_schema().names())
    missing = REQUIRED_COLUMNS - schema_cols
    if missing:
        raise ValueError(
            f"{path} is missing CICIoV2024 columns: {sorted(missing)}"
        )

    can_id_expr = (
        pl.col("ID")
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace("^0X", "")
        .str.replace(r"\.[0-9]+$", "")
        .str.strip_chars_start("0")
        .str.zfill(3)
        .alias("Arbitration_ID")
    )
    atk_expr = (
        pl.when(pl.col("specific_class").str.strip_chars().str.to_uppercase() == "BENIGN")
        .then(pl.lit("BENIGN"))
        .when(pl.col("specific_class").str.strip_chars().str.to_uppercase() == "DOS")
        .then(pl.lit("DoS"))
        .otherwise(pl.col("specific_class").str.strip_chars().str.to_uppercase())
        .alias("attack_type")
    )
    if "DLC" in schema_cols:
        dlc_expr = (
            pl.when(
                pl.col("DLC").is_null()
                | (pl.col("DLC").str.strip_chars("[] \t\r\n") == "")
            )
            .then(pl.lit(8, dtype=pl.UInt8))
            .otherwise(
                pl.col("DLC")
                .str.strip_chars("[] \t\r\n")
                .str.replace(r"\.[0-9]+$", "")
                .cast(pl.UInt8)
            )
            .alias("DLC")
        )
    else:
        dlc_expr = pl.lit(8, dtype=pl.UInt8).alias("DLC")

    data_exprs = [
        pl.when(
            pl.col(f"DATA_{i}").is_null()
            | (pl.col(f"DATA_{i}").str.strip_chars() == "")
        )
        .then(pl.lit("PAD"))
        .otherwise(
            pl.col(f"DATA_{i}")
            .str.strip_chars()
            .str.to_uppercase()
            .str.replace("^0X", "")
            .str.replace(r"\.[0-9]+$", "")
            .str.zfill(2)
        )
        .alias(f"Data_{i}")
        for i in range(8)
    ]

    transformed = lf.with_columns(
        pl.lit(path.stem).alias("session_id"),
        can_id_expr,
        atk_expr,
        dlc_expr,
        *data_exprs,
    ).select(
        "session_id",
        "Arbitration_ID",
        "DLC",
        *[f"Data_{i}" for i in range(8)],
        "attack_type",
    )
    yield from transformed.collect_batches()


def read(files: list[Path]) -> pl.DataFrame:
    """Compatibility reader for small tests; production uses ``write``."""
    frames = [chunk for path in files for chunk in _iter_chunks(path)]
    if not frames:
        raise ValueError("CICIoV2024 adapter received no valid frames")
    return pl.concat(frames, how="vertical")


def write(files: list[Path], output: Path) -> int:
    """Stream all selected CICIoV2024 files into one canonical Parquet file."""
    if not files:
        raise ValueError("CICIoV2024 adapter received no files")
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
            print(f"Ingested CICIoV2024 {path.name}: {file_rows:,} frames")
        if writer is None:
            raise ValueError("CICIoV2024 adapter found no valid frames")
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(temporary_path, output)
    return total_rows
