"""Adapter for CICIoV2024 hexadecimal CSV files."""

from pathlib import Path

import polars as pl


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


def read(files: list[Path]) -> pl.DataFrame:
    frames = []
    for path in files:
        frame = pl.read_csv(path, infer_schema_length=0).with_columns(
            pl.all().cast(pl.String)
        )
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path} is missing CICIoV2024 columns: {sorted(missing)}"
            )

        # CICIoV2024 has no physical timestamp. Preserve source order through
        # row_id; prepare.py will only create timing features when possible.
        expressions = [
            pl.lit(path.stem).alias("session_id"),
            pl.col("ID")
            .map_elements(_can_id, return_dtype=pl.String)
            .alias("Arbitration_ID"),
            pl.col("specific_class")
            .map_elements(_attack_type, return_dtype=pl.String)
            .alias("attack_type"),
            *[
                pl.col(column)
                .map_elements(_hex_byte, return_dtype=pl.String)
                .alias(f"Data_{index}")
                for index, column in enumerate(DATA_COLUMNS)
            ],
        ]
        if "DLC" in frame.columns:
            expressions.append(
                pl.col("DLC").map_elements(_dlc, return_dtype=pl.UInt8).alias("DLC")
            )
        else:
            expressions.append(pl.lit(8, dtype=pl.UInt8).alias("DLC"))
        frame = frame.with_columns(expressions)
        frames.append(
            frame.select(
                "session_id",
                "Arbitration_ID",
                "DLC",
                *[f"Data_{i}" for i in range(8)],
                "attack_type",
            )
        )

    return pl.concat(frames, how="vertical")
