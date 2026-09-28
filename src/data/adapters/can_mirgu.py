"""Streaming adapter for the CAN-MIRGU CAN dump files.

The final integer on each log line is the source frame flag: ``1`` means an
injected frame and ``0`` means a normal frame. The adapter keeps that source
label and only uses the filename/directory to assign the attack family.
"""

import os
import re
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


LOG_LINE = re.compile(
    r"^\((?P<timestamp>[^)]+)\)\s+\S+\s+"
    r"(?P<arbitration_id>[0-9A-Fa-f]+)#(?P<payload>[0-9A-Fa-f]*)\s+"
    r"(?P<flag>[01])\s*$"
)
CHUNK_SIZE = 100_000
ATTACK_FAMILIES = {"real", "extended"}


def _can_id(value):
    """Canonicalize equivalent hexadecimal IDs to one three-digit value."""
    text = str(value).strip().upper().removeprefix("0X")
    if not text:
        raise ValueError("CAN-MIRGU CAN ID cannot be empty")
    number = int(text, 16)
    if not 0 <= number <= 0x7FF:
        raise ValueError(f"CAN-MIRGU CAN ID is outside the 11-bit range: {value!r}")
    return f"{number:03X}"


def _payload(value):
    text = str(value).strip().upper()
    if len(text) % 2 or len(text) > 16:
        raise ValueError(f"Invalid CAN-MIRGU payload: {value!r}")
    if not text:
        text = "0"
    int(text, 16)
    return text.zfill(16)


def _payload_bytes(payload):
    return [payload[index:index + 2] for index in range(0, 16, 2)]


def _attack_family(path: Path):
    relative_parts = {part.lower() for part in path.parts}
    stem = path.stem.lower()
    if "benign" in relative_parts:
        return "Normal"
    if "masquerade_attacks" in relative_parts:
        return "Masquerade"
    if "suspension_attacks" in relative_parts:
        return "Suspension"
    if "fuzzing" in stem:
        return "Fuzzing"
    if "dos" in stem:
        return "DoS"
    if "replay" in stem:
        return "Replay"
    if "real_attacks" in relative_parts:
        return "Spoofing"
    raise ValueError(f"Cannot infer CAN-MIRGU family from path: {path}")


def _frame_attack_type(
    stem: str, family: str, can_id: str, rel_timestamp: float
) -> str:
    if stem == "fuzzing_valid_ids_dos":
        return "DoS" if can_id == "000" else "Fuzzing"
    if stem == "reverse_speedometer_fuzzing_attack":
        return "Spoofing" if rel_timestamp < 70.0 else "Fuzzing"
    if stem == "multiple_attacks_2":
        return "Fuzzing" if rel_timestamp >= 880.0 else "Spoofing"
    return family


def _iter_chunks(path: Path, chunk_size=CHUNK_SIZE):
    family = _attack_family(path)
    stem = path.stem.lower()
    rows = []
    malformed = 0
    first_timestamp = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            match = LOG_LINE.match(text)
            if match is None:
                malformed += 1
                continue
            timestamp = float(match.group("timestamp"))
            if first_timestamp is None:
                first_timestamp = timestamp
            rel_timestamp = timestamp - first_timestamp
            can_id = _can_id(match.group("arbitration_id"))
            payload = _payload(match.group("payload"))
            flag = int(match.group("flag"))
            rows.append(
                {
                    "session_id": path.stem,
                    "Timestamp": rel_timestamp,
                    "Arbitration_ID": can_id,
                    "DLC": len(match.group("payload")) // 2,
                    **{
                        f"Data_{index}": value
                        for index, value in enumerate(_payload_bytes(payload))
                    },
                    "attack_type": (
                        "Normal"
                        if family == "Normal" or flag == 0
                        else _frame_attack_type(stem, family, can_id, rel_timestamp)
                    ),
                }
            )
            if len(rows) >= chunk_size:
                yield rows
                rows = []

    if malformed:
        print(f"Skipped {malformed:,} malformed CAN-MIRGU lines in {path}")
    if rows:
        yield rows


def select_files(files: list[Path], variant: str | None):
    """Select the requested CAN-MIRGU corpus without mixing attack families."""
    if variant not in ATTACK_FAMILIES:
        raise ValueError("CAN-MIRGU requires dataset.variant='real' or 'extended'")

    selected = []
    for path in files:
        family = _attack_family(path)
        if family is None:
            continue
        if variant == "real" and family in {"Masquerade", "Suspension"}:
            continue
        selected.append(path)
    return sorted(selected)


def read(files: list[Path]) -> pl.DataFrame:
    """Compatibility reader for small tests; production uses ``write``."""
    frames = [pl.DataFrame(rows) for path in files for rows in _iter_chunks(path)]
    if not frames:
        raise ValueError("CAN-MIRGU adapter received no valid log rows")
    return pl.concat(frames, how="vertical").with_columns(
        pl.col("DLC").cast(pl.UInt8),
    )


def write(files: list[Path], output: Path) -> int:
    """Stream all selected CAN-MIRGU files into one canonical Parquet file."""
    if not files:
        raise ValueError("CAN-MIRGU adapter received no files")
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
            for rows in _iter_chunks(path):
                frame = pl.DataFrame(rows).with_columns(pl.col("DLC").cast(pl.UInt8))
                frame.insert_column(
                    0,
                    pl.Series(
                        "row_id",
                        range(total_rows, total_rows + len(frame)),
                        dtype=pl.UInt64,
                    ),
                )
                table = frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                total_rows += len(frame)
                file_rows += len(frame)
            print(f"Ingested CAN-MIRGU {path.name}: {file_rows:,} frames")
        if writer is None:
            raise ValueError("CAN-MIRGU adapter found no valid frames")
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(temporary_path, output)
    return total_rows
