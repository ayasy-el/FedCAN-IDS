"""Adapter for the Survival Analysis Dataset CAN logs."""

import csv
from pathlib import Path

import polars as pl


MAX_COLUMNS = 12
RAW_COLUMNS = [f"raw_{index}" for index in range(MAX_COLUMNS)]
ATTACK_TYPES = {
    "freedriving": "Normal",
    "flooding": "Flooding",
    "fuzzy": "Fuzzing",
    "malfunction": "Malfunction",
}


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
    """Validate and canonicalize an 11-bit CAN ID to three hex digits."""
    text = str(value).strip().upper().removeprefix("0X")
    if not text:
        raise ValueError("CAN ID cannot be empty")
    try:
        number = int(text, 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hexadecimal CAN ID: {value!r}") from exc
    if not 0 <= number <= 0x7FF:
        raise ValueError(f"CAN ID is outside the 11-bit range: {value!r}")
    return f"{number:03X}"


def _parse_rows(path: Path) -> pl.DataFrame:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if len(row) > MAX_COLUMNS:
                raise ValueError(
                    f"{path}:{line_number} has {len(row)} columns; "
                    f"expected at most {MAX_COLUMNS}"
                )
            rows.append(row + [None] * (MAX_COLUMNS - len(row)))
    return pl.DataFrame(rows, schema=RAW_COLUMNS, orient="row")


def _attack_type(path: Path) -> str:
    stem = path.stem.lower()
    for marker, label in ATTACK_TYPES.items():
        if marker in stem:
            return label
    raise ValueError(
        f"Cannot infer Survival Analysis attack type from filename: {path.name}"
    )


def read(files: list[Path]) -> pl.DataFrame:
    frames = []
    for path in files:
        raw = _parse_rows(path)
        scenario = _attack_type(path)
        data_fields = [f"raw_{index}" for index in range(3, 11)]
        # Some logs store the payload as one space-separated field; others use
        # one CSV field per byte. Normalize both forms to eight byte columns.
        payload = pl.when(
            pl.col("raw_3").fill_null("").str.contains(r"\s")
        ).then(
            pl.col("raw_3").fill_null("").str.split(" ")
        ).otherwise(
            pl.concat_list(data_fields).list.drop_nulls()
        )
        flag = (
            pl.concat_list([f"raw_{index}" for index in range(3, 12)])
            .list.drop_nulls()
            .list.last()
            .cast(pl.String)
            .str.strip_chars()
            .str.to_uppercase()
        )
        frame = raw.with_columns(
            pl.col("raw_0").cast(pl.Float64).alias("Timestamp"),
            pl.col("raw_1")
            .map_elements(_can_id, return_dtype=pl.String)
            .alias("Arbitration_ID"),
            pl.col("raw_2")
            .cast(pl.UInt8)
            .alias("DLC"),
            payload.alias("Payload"),
            flag.alias("Flag"),
        ).with_columns(
            *[
                pl.col("Payload")
                .list.get(index, null_on_oob=True)
                .map_elements(_hex_byte, return_dtype=pl.String)
                .alias(f"Data_{index}")
                for index in range(8)
            ],
            pl.lit(path.stem).alias("session_id"),
            pl.when((pl.col("Flag") == "T") & (pl.lit(scenario) != "Normal"))
            .then(pl.lit(scenario))
            .otherwise(pl.lit("Normal"))
            .alias("attack_type"),
        )
        frames.append(
            frame.select(
                "session_id",
                "Timestamp",
                "Arbitration_ID",
                "DLC",
                *[f"Data_{i}" for i in range(8)],
                "attack_type",
            )
        )
    return pl.concat(frames, how="vertical")
