"""Adapter for the five-scenario Car-Hacking Dataset.

The attack logs are ragged CSV files with a trailing T/R flag. The normal log
is a text file with labelled fields. In both formats, the flag is authoritative
for the frame label: T is injected traffic and R is normal traffic.
"""

import re
from pathlib import Path

import polars as pl


MAX_COLUMNS = 12
RAW_COLUMNS = [f"raw_{index}" for index in range(MAX_COLUMNS)]
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


def _flag_expression():
    return (
        pl.concat_list([f"raw_{index}" for index in range(3, MAX_COLUMNS)])
        .list.drop_nulls()
        .list.last()
        .cast(pl.String)
        .str.strip_chars()
        .str.to_uppercase()
    )


def _read_attack_csv(path: Path) -> pl.DataFrame:
    raw = pl.read_csv(
        path,
        has_header=False,
        new_columns=RAW_COLUMNS,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    scenario = _scenario(path)
    payload = pl.concat_list(
        [f"raw_{index}" for index in range(3, 11)]
    ).list.drop_nulls()
    return (
        raw.with_columns(
            pl.col("raw_0").cast(pl.Float64).alias("Timestamp"),
            pl.col("raw_1")
            .map_elements(_can_id, return_dtype=pl.String)
            .alias("Arbitration_ID"),
            pl.col("raw_2").cast(pl.UInt8).alias("DLC"),
            payload.alias("Payload"),
            _flag_expression().alias("Flag"),
        )
        .with_columns(
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
        .select(
            "session_id",
            "Timestamp",
            "Arbitration_ID",
            "DLC",
            *[f"Data_{index}" for index in range(8)],
            "attack_type",
        )
    )


def _read_normal_txt(path: Path) -> pl.DataFrame:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = NORMAL_LINE.match(line.strip())
            if match is None:
                # The distributed file has one incomplete trailing line.
                if line.strip():
                    continue
                continue
            timestamp, arbitration_id, dlc, payload = match.groups()
            bytes_ = payload.split()[:8]
            rows.append(
                {
                    "session_id": path.stem,
                    "Timestamp": float(timestamp),
                    "Arbitration_ID": _can_id(arbitration_id),
                    "DLC": int(dlc),
                    **{
                        f"Data_{index}": _hex_byte(
                            bytes_[index] if index < len(bytes_) else None
                        )
                        for index in range(8)
                    },
                    "attack_type": "Normal",
                }
            )
    if not rows:
        raise ValueError(f"No valid CAN frames found in {path}")
    return pl.DataFrame(rows).with_columns(pl.col("DLC").cast(pl.UInt8))


def read(files: list[Path]) -> pl.DataFrame:
    frames = []
    for path in files:
        if path.suffix.lower() == ".csv":
            frames.append(_read_attack_csv(path))
        elif path.suffix.lower() == ".txt":
            frames.append(_read_normal_txt(path))
        else:
            raise ValueError(f"Unsupported Car-Hacking file format: {path}")
    return pl.concat(frames, how="vertical")
