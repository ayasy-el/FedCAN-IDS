"""Streaming adapter for the Survival Analysis Dataset CAN logs."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


CHUNK_SIZE = 100_000
ATTACK_TYPES = {
    "freedriving": "Normal",
    "flooding": "Flooding",
    "fuzzy": "Fuzzing",
    "malfunction": "Malfunction",
}

HEX_LOOKUP = {f"{i:02x}": f"{i:02X}" for i in range(256)}
HEX_LOOKUP.update({f"{i:02X}": f"{i:02X}" for i in range(256)})
HEX_LOOKUP.update({f"{i:x}": f"{i:02X}" for i in range(256)})
HEX_LOOKUP.update({f"{i:X}": f"{i:02X}" for i in range(256)})

CAN_ID_LOOKUP = {f"{i:03x}": f"{i:03X}" for i in range(0x800)}
CAN_ID_LOOKUP.update({f"{i:03X}": f"{i:03X}" for i in range(0x800)})
CAN_ID_LOOKUP.update({f"{i:04x}": f"{i:03X}" for i in range(0x800)})
CAN_ID_LOOKUP.update({f"{i:04X}": f"{i:03X}" for i in range(0x800)})
CAN_ID_LOOKUP.update({f"{i:x}": f"{i:03X}" for i in range(0x800)})
CAN_ID_LOOKUP.update({f"{i:X}": f"{i:03X}" for i in range(0x800)})


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


def _attack_type(path: Path) -> str:
    stem = path.stem.lower()
    for marker, label in ATTACK_TYPES.items():
        if marker in stem:
            return label
    raise ValueError(
        f"Cannot infer Survival Analysis attack type from filename: {path.name}"
    )


def _iter_chunks(path: Path, chunk_size=CHUNK_SIZE):
    scenario = _attack_type(path)
    session_id = path.stem
    ts_list, id_list, dlc_list = [], [], []
    d0, d1, d2, d3, d4, d5, d6, d7 = [], [], [], [], [], [], [], []
    atk_list = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                ts = float(parts[0])
                cid_raw = parts[1].strip()
                cid = CAN_ID_LOOKUP.get(cid_raw) or f"{int(cid_raw, 16):03X}"
                dlc = int(parts[2])
            except ValueError:
                continue

            p3 = parts[3].strip()
            if " " in p3:
                bytes_ = p3.split()[:8]
                flag = "R"
            else:
                last_part = parts[-1].strip().upper()
                if last_part in ("R", "T"):
                    flag = last_part
                    bytes_ = parts[3:-1]
                else:
                    flag = "R"
                    bytes_ = parts[3:]

            ts_list.append(ts)
            id_list.append(cid)
            dlc_list.append(dlc)
            atk_list.append(
                scenario if flag == "T" and scenario != "Normal" else "Normal"
            )

            n = len(bytes_)
            d0.append(HEX_LOOKUP.get(bytes_[0].strip(), "PAD") if n > 0 else "PAD")
            d1.append(HEX_LOOKUP.get(bytes_[1].strip(), "PAD") if n > 1 else "PAD")
            d2.append(HEX_LOOKUP.get(bytes_[2].strip(), "PAD") if n > 2 else "PAD")
            d3.append(HEX_LOOKUP.get(bytes_[3].strip(), "PAD") if n > 3 else "PAD")
            d4.append(HEX_LOOKUP.get(bytes_[4].strip(), "PAD") if n > 4 else "PAD")
            d5.append(HEX_LOOKUP.get(bytes_[5].strip(), "PAD") if n > 5 else "PAD")
            d6.append(HEX_LOOKUP.get(bytes_[6].strip(), "PAD") if n > 6 else "PAD")
            d7.append(HEX_LOOKUP.get(bytes_[7].strip(), "PAD") if n > 7 else "PAD")

            if len(ts_list) >= chunk_size:
                yield pl.DataFrame({
                    "session_id": [session_id] * len(ts_list),
                    "Timestamp": ts_list,
                    "Arbitration_ID": id_list,
                    "DLC": pl.Series(dlc_list, dtype=pl.UInt8),
                    "Data_0": d0, "Data_1": d1, "Data_2": d2, "Data_3": d3,
                    "Data_4": d4, "Data_5": d5, "Data_6": d6, "Data_7": d7,
                    "attack_type": atk_list,
                })
                ts_list, id_list, dlc_list = [], [], []
                d0, d1, d2, d3, d4, d5, d6, d7 = [], [], [], [], [], [], [], []
                atk_list = []

    if ts_list:
        yield pl.DataFrame({
            "session_id": [session_id] * len(ts_list),
            "Timestamp": ts_list,
            "Arbitration_ID": id_list,
            "DLC": pl.Series(dlc_list, dtype=pl.UInt8),
            "Data_0": d0, "Data_1": d1, "Data_2": d2, "Data_3": d3,
            "Data_4": d4, "Data_5": d5, "Data_6": d6, "Data_7": d7,
            "attack_type": atk_list,
        })


def read(files: list[Path]) -> pl.DataFrame:
    """Compatibility reader for small tests; production uses ``write``."""
    frames = [chunk for path in files for chunk in _iter_chunks(path)]
    if not frames:
        raise ValueError("Survival Analysis adapter received no valid frames")
    return pl.concat(frames, how="vertical")


def write(files: list[Path], output: Path) -> int:
    """Stream all selected Survival Analysis files into one canonical Parquet file."""
    if not files:
        raise ValueError("Survival Analysis adapter received no files")
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
            print(f"Ingested Survival Analysis {path.name}: {file_rows:,} frames")
        if writer is None:
            raise ValueError("Survival Analysis adapter found no valid frames")
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        os.replace(temporary_path, output)
    return total_rows
