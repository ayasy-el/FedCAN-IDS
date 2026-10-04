"""Streaming adapter for the five-scenario Car-Hacking Dataset.

The attack logs are ragged CSV files with a trailing T/R flag. The normal log
is a text file with labelled fields. In both formats, the flag is authoritative
for the frame label: T is injected traffic and R is normal traffic.
"""

import os
import re
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


MAX_COLUMNS = 12
RAW_COLUMNS = [f"raw_{index}" for index in range(MAX_COLUMNS)]
CHUNK_SIZE = 100_000
SCENARIOS = {
    "dos": "DoS",
    "fuzzy": "Fuzzy",
    "gear": "Gear",
    "rpm": "RPM",
}
NORMAL_LINE = re.compile(
    r"Timestamp:\s*(\S+)\s+ID:\s*(\S+)\s+\S+\s+DLC:\s*(\d+)\s*(.*)$"
)


def _hex_byte(value):
    if value is None or not str(value).strip():
        return "PAD"
    text = str(value).strip().upper().removeprefix("0X")
    if text in {"R", "T"}:
        return "PAD"
    number = int(text, 16)
    if not 0 <= number <= 255:
        raise ValueError(f"Invalid CAN byte value: {value!r}")
    return f"{number:02X}"


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


def _scenario(path: Path):
    stem = path.stem.lower()
    for marker, label in SCENARIOS.items():
        if marker in stem:
            return label
    return "Normal"


def _iter_attack_csv_chunks(path: Path):
    scenario = _scenario(path)
    lf = pl.scan_csv(
        path,
        has_header=False,
        new_columns=RAW_COLUMNS,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    can_id_expr = (
        pl.col("raw_1")
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace("^0X", "")
        .str.strip_chars_start("0")
        .str.zfill(3)
        .alias("Arbitration_ID")
    )
    flag_expr = (
        pl.concat_list([f"raw_{index}" for index in range(3, MAX_COLUMNS)])
        .list.drop_nulls()
        .list.last()
        .str.strip_chars()
        .str.to_uppercase()
        .alias("Flag")
    )
    dlc_expr = pl.col("raw_2").cast(pl.UInt8).alias("DLC")
    byte_exprs = [
        pl.when(pl.col("raw_2").cast(pl.UInt8) > i)
        .then(
            pl.col(f"raw_{i + 3}")
            .str.strip_chars()
            .str.to_uppercase()
            .str.replace("^0X", "")
            .str.zfill(2)
        )
        .otherwise(pl.lit("PAD"))
        .alias(f"Data_{i}")
        for i in range(8)
    ]
    transformed = (
        lf.with_columns(
            pl.lit(path.stem).alias("session_id"),
            pl.col("raw_0").cast(pl.Float64).alias("Timestamp"),
            can_id_expr,
            dlc_expr,
            flag_expr,
        )
        .with_columns(byte_exprs)
        .with_columns(
            pl.when((pl.col("Flag") == "T") & (pl.lit(scenario) != "Normal"))
            .then(pl.lit(scenario))
            .otherwise(pl.lit("Normal"))
            .alias("attack_type")
        )
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


def _iter_normal_txt_chunks(path: Path, chunk_size=CHUNK_SIZE):
    session_id = path.stem
    ts_list, id_list, dlc_list = [], [], []
    d_lists = [[] for _ in range(8)]
    atk_list = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            match = NORMAL_LINE.match(text)
            if match is None:
                continue
            timestamp, arbitration_id, dlc_str, payload = match.groups()
            ts_list.append(float(timestamp))
            id_list.append(f"{int(arbitration_id, 16):03X}")
            dlc = int(dlc_str)
            dlc_list.append(dlc)
            bytes_ = payload.split()[:8]
            for i in range(8):
                if i < len(bytes_):
                    d_lists[i].append(f"{int(bytes_[i], 16):02X}")
                else:
                    d_lists[i].append("PAD")
            atk_list.append("Normal")

            if len(ts_list) >= chunk_size:
                yield pl.DataFrame({
                    "session_id": [session_id] * len(ts_list),
                    "Timestamp": ts_list,
                    "Arbitration_ID": id_list,
                    "DLC": pl.Series(dlc_list, dtype=pl.UInt8),
                    **{f"Data_{i}": d_lists[i] for i in range(8)},
                    "attack_type": atk_list,
                })
                ts_list, id_list, dlc_list = [], [], []
                d_lists = [[] for _ in range(8)]
                atk_list = []

    if ts_list:
        yield pl.DataFrame({
            "session_id": [session_id] * len(ts_list),
            "Timestamp": ts_list,
            "Arbitration_ID": id_list,
            "DLC": pl.Series(dlc_list, dtype=pl.UInt8),
            **{f"Data_{i}": d_lists[i] for i in range(8)},
            "attack_type": atk_list,
        })


def _iter_chunks(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from _iter_attack_csv_chunks(path)
    elif suffix == ".txt":
        yield from _iter_normal_txt_chunks(path)
    else:
        raise ValueError(f"Unsupported Car-Hacking file format: {path}")


def read(files: list[Path]) -> pl.DataFrame:
    """Compatibility reader for small tests; production uses ``write``."""
    frames = [chunk for path in files for chunk in _iter_chunks(path)]
    if not frames:
        raise ValueError("Car-Hacking adapter received no valid frames")
    return pl.concat(frames, how="vertical")


def write(files: list[Path], output: Path) -> int:
    """Stream all selected Car-Hacking files into one canonical Parquet file."""
    if not files:
        raise ValueError("Car-Hacking adapter received no files")
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
            print(f"Ingested Car-Hacking {path.name}: {file_rows:,} frames")
        if writer is None:
            raise ValueError("Car-Hacking adapter found no valid frames")
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(temporary_path, output)
    return total_rows
