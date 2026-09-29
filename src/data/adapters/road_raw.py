"""Adapter for the raw ORNL ROAD CAN logs.

ROAD ``.log`` files do not contain frame labels. Labels are reconstructed from
the capture metadata, injection interval, target CAN ID, and attack payload
pattern. This follows the reference ``add_actual_attack_col`` logic supplied
with the ROAD reproduction code. The paper preprocessing also removes frames
with CAN IDs greater than ``0x700``; this adapter applies that cutoff before
labeling and writing the canonical data.
"""

import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


LOG_LINE = re.compile(
    r"^\((?P<timestamp>[^)]+)\)\s+\S+\s+"
    r"(?P<arbitration_id>[0-9A-Fa-f]+)#(?P<payload>[0-9A-Fa-f]*)"
)
ATTACK_TYPES = (
    ("correlated_signal", "CS"),
    ("fuzzing", "Fuzzing"),
    ("max_engine_coolant_temp", "MEC"),
    ("max_speedometer", "MS"),
    ("reverse_light_on", "RLOn"),
    ("reverse_light_off", "RLOff"),
)
CHUNK_SIZE = 100_000
MAX_CAN_ID = 0x700


def _can_id(value):
    text = str(value).strip().upper().removeprefix("0X")
    number = int(text, 16)
    if not 0 <= number <= 0xFFF:
        raise ValueError(f"CAN ID is outside the supported ROAD range: {value!r}")
    return f"{number:03X}"


def _payload(value):
    text = str(value).strip().upper()
    if len(text) % 2 or not text:
        raise ValueError(f"Invalid ROAD payload: {value!r}")
    int(text, 16)
    if len(text) > 16:
        raise ValueError(f"ROAD payload exceeds 8 bytes: {value!r}")
    # Match the supplied reference preprocessing: missing bytes are left-padded.
    return text.zfill(16)


def _payload_bytes(payload):
    return [payload[index:index + 2] for index in range(0, 16, 2)]


def _attack_name(path: Path):
    stem = path.stem.lower()
    for prefix, label in ATTACK_TYPES:
        if stem.startswith(prefix):
            return label
    return None


def _metadata_path(path: Path):
    candidate = path.parent / "capture_metadata.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"ROAD metadata not found beside {path}: {candidate}")


@lru_cache(maxsize=4)
def _load_metadata(path_string):
    with Path(path_string).open(encoding="utf-8") as handle:
        return json.load(handle)


def _capture_metadata(path: Path):
    metadata = _load_metadata(str(_metadata_path(path)))
    return metadata.get(path.stem, {})


def _matches_payload_pattern(payload: str, pattern: str) -> bool:
    if len(payload) != len(pattern):
        return False
    return all(p == pat for p, pat in zip(payload, pattern) if pat != "X")


def _matches_reference_rule(attack_name, metadata, relative_time, aid, payload):
    interval = metadata.get("injection_interval")
    pattern = metadata.get("injection_data_str")
    target = metadata.get("injection_id")
    if not interval or not pattern:
        return False
    if not interval[0] <= round(relative_time, 6) <= interval[1]:
        return False

    pattern = pattern.upper()
    target = str(target).upper() if target is not None else None
    if target and target != "XXX" and aid != _can_id(target):
        return False

    return _matches_payload_pattern(payload, pattern)


def _iter_log_chunks(path: Path, chunk_size: int = CHUNK_SIZE):
    """Yield bounded batches of canonical rows from one ROAD log."""
    metadata = _capture_metadata(path)
    attack_name = _attack_name(path)
    rows = []
    malformed = 0
    out_of_scope = 0
    first_timestamp = None

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_LINE.match(line.strip())
            if match is None:
                if line.strip():
                    malformed += 1
                continue
            timestamp = float(match.group("timestamp"))
            if first_timestamp is None:
                first_timestamp = timestamp
            relative_time = timestamp - first_timestamp
            aid = _can_id(match.group("arbitration_id"))
            if int(aid, 16) > MAX_CAN_ID:
                out_of_scope += 1
                continue
            payload = _payload(match.group("payload"))
            injected = bool(
                attack_name
                and _matches_reference_rule(
                    attack_name,
                    metadata,
                    relative_time,
                    aid,
                    payload,
                )
            )
            rows.append(
                {
                    "session_id": path.stem,
                    "Timestamp": relative_time,
                    "Arbitration_ID": aid,
                    "DLC": len(match.group("payload")) // 2,
                    **{
                        f"Data_{index}": value
                    for index, value in enumerate(_payload_bytes(payload))
                    },
                    "attack_type": attack_name if injected else "Normal",
                }
            )
            if len(rows) >= chunk_size:
                yield rows
                rows = []

    if not rows:
        if malformed and first_timestamp is None:
            raise ValueError(f"No valid ROAD CAN frames found in {path}")
    if malformed:
        print(f"Skipped {malformed:,} malformed ROAD lines in {path}")
    if out_of_scope:
        print(
            f"Skipped {out_of_scope:,} out-of-scope ROAD frames "
            f"(CAN ID > 0x{MAX_CAN_ID:X}) in {path}"
        )
    if rows:
        yield rows


def _read_log(path: Path) -> pl.DataFrame:
    """Read one log as a DataFrame for small-file tests and compatibility."""
    batches = [pl.DataFrame(rows) for rows in _iter_log_chunks(path)]
    if not batches:
        raise ValueError(f"No valid ROAD CAN frames found in {path}")
    return pl.concat(batches, how="vertical").with_columns(
        pl.col("DLC").cast(pl.UInt8),
    )


def select_files(files: list[Path], variant: str | None) -> list[Path]:
    """Select one paper experiment without mixing fabrication and masquerade."""
    if variant not in {"fabrication", "masquerade"}:
        raise ValueError(
            "ROAD requires dataset.variant='fabrication' or 'masquerade'"
        )

    selected = []
    for path in files:
        if path.parent.name == "ambient":
            # selected.append(path)
            continue
        if path.parent.name != "attacks":
            continue
        if path.stem.startswith("accelerator_attack_"):
            continue
        is_masquerade = path.stem.endswith("_masquerade")
        if (variant == "masquerade") == is_masquerade:
            selected.append(path)
    return sorted(selected)


def read(files: list[Path]) -> pl.DataFrame:
    if not files:
        raise ValueError("ROAD adapter received no log files")
    return pl.concat([_read_log(path) for path in files], how="vertical")


def write(files: list[Path], output: Path) -> int:
    """Stream ROAD logs into one Parquet file without materializing all rows."""
    if not files:
        raise ValueError("ROAD adapter received no log files")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()

    writer = None
    total_rows = 0
    try:
        for path in files:
            file_rows = 0
            for rows in _iter_log_chunks(path):
                frame = pl.DataFrame(rows).with_columns(
                    pl.col("DLC").cast(pl.UInt8),
                )
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
                        temporary_path,
                        table.schema,
                        compression="zstd",
                    )
                writer.write_table(table)
                total_rows += len(frame)
                file_rows += len(frame)
            if file_rows:
                print(f"Ingested ROAD {path.name}: {file_rows:,} frames")

        if writer is None:
            raise ValueError("ROAD adapter found no valid CAN frames")
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(temporary_path, output)

    return total_rows
