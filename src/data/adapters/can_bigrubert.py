"""Faithful CAN-BiGRUBERT data adapter."""

from pathlib import Path

import numpy as np
import polars as pl


ATTACKS = {"Normal": 0, "Flooding": 1, "Fuzzing": 2, "Spoofing": 3, "Replay": 4}
STATE_OFFSET = {"D": 0, "S": 5}
CLASS_NAMES = [
    "benign-driving", "DoS-driving", "Fuzzing-driving", "Spoofing-driving", "Replay-driving",
    "benign-stationary", "DoS-stationary", "Fuzzing-stationary", "Spoofing-stationary", "Replay-stationary",
]


def _state_from_name(name):
    if "_D_" in name or name.endswith("_D"):
        return "D"
    if "_S_" in name or name.endswith("_S"):
        return "S"
    raise ValueError(f"Cannot infer vehicle state from log name: {name}")


def _read_raw(params):
    dataset = params["dataset"]
    raw_dir = Path(dataset["raw_dir"])
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    return files


def _canonical_frame_data(params):
    frames = []
    for path in sorted(_read_raw(params), key=lambda p: p.name):
        state = _state_from_name(path.stem)
        df = pl.read_csv(path, infer_schema_length=1000)
        if "SubClass" not in df.columns:
            df = df.with_columns(pl.lit("Normal").alias("SubClass"))
        df = (
            df.with_columns(
                pl.lit(path.stem).alias("session_id"),
                pl.lit(state).alias("vehicle_state"),
                pl.col("DLC").cast(pl.UInt8),
            )
            .with_columns(
                pl.col("Data")
                .str.split(" ")
                .list.to_struct(fields=[f"Data_{i}" for i in range(8)])
                .alias("Bytes")
            )
            .unnest("Bytes")
            .with_columns(
                *[pl.col(f"Data_{i}").fill_null("PAD") for i in range(8)]
            )
            .with_columns(
                pl.col("SubClass").replace_strict(ATTACKS).cast(pl.UInt8).alias("attack_id")
            )
            .with_columns(
                (pl.col("attack_id") + (5 if state == "S" else 0))
                .cast(pl.UInt8)
                .alias("paper_class")
            )
            .drop("Class", "SubClass", "Data", strict=False)
            .rename({"paper_class": "Class"})
        )
        frames.append(df)
    return pl.concat(frames, how="diagonal").with_row_index("row_id")


def ingest(params):
    dataset = params["dataset"]
    paths = dataset["sequence_window"]
    output_path = Path(paths["interim_path"])
    df = _canonical_frame_data(params)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    print(df.head())


def split(params):
    dataset = params["dataset"]
    profile = params["data"]["sequence_window"]
    paths = dataset["sequence_window"]
    output_dir = Path(paths["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pl.read_parquet(paths["interim_path"])
    balanced = _balanced_windows(
        source,
        int(profile["window_size"]),
        int(profile["stride"]),
        int(profile["balance_size"]),
        int(profile["random_seed"]),
    )
    splits = _split_balanced_windows(balanced, int(profile["random_seed"]))
    for split_name, records in splits.items():
        pl.DataFrame(records).write_parquet(
            output_dir / f"{split_name}.parquet", compression="zstd"
        )


def _frame_string(row):
    # Paper does not specify prefix, casing, or DLC formatting. This is the
    # explicit implementation choice used by this reproduction: uppercase
    # hexadecimal components without a 0x prefix, whitespace-separated.
    aid = str(row["Arbitration_ID"]).upper().removeprefix("0X")
    dlc = format(int(row["DLC"]), "X")
    values = [aid, dlc]
    values.extend(str(row[f"Data_{i}"]).upper() for i in range(8))
    return " ".join(str(value) for value in values)


def _candidate_starts(labels, target_label, window_size, stride):
    """Return valid window starts for one attack/state class."""
    labels = np.asarray(labels, dtype=np.int16)
    n_rows = len(labels)
    n_starts = n_rows - window_size + 1
    if n_starts <= 0:
        return np.empty(0, dtype=np.int64)
    attack = (labels != 0) & (labels != 5)
    if target_label in (0, 5):
        prefix = np.concatenate(([0], np.cumsum(attack, dtype=np.int64)))
        valid = (prefix[window_size:] - prefix[:-window_size]) == 0
    else:
        target = labels == target_label
        prefix = np.concatenate(([0], np.cumsum(target, dtype=np.int64)))
        valid = (prefix[window_size:] - prefix[:-window_size]) > 0
    return np.flatnonzero(valid)[::stride]


def _balanced_windows(df, window_size, stride, target, seed):
    """Sample balanced windows without materializing all candidate windows."""
    reservoirs = {label: [] for label in range(10)}
    seen = {label: 0 for label in range(10)}
    rng = np.random.default_rng(seed)
    for session in df.partition_by("session_id", maintain_order=True):
        labels = session["Class"].to_numpy()
        state = session["vehicle_state"][0]
        benign_label = 0 if state == "D" else 5
        for label in range(10):
            effective_label = benign_label if label in (0, 5) else label
            starts = _candidate_starts(labels, effective_label, window_size, stride)
            if effective_label == benign_label and label != benign_label:
                starts = np.empty(0, dtype=np.int64)
            for start in starts:
                seen[label] += 1
                item = (session, int(start))
                if len(reservoirs[label]) < target:
                    reservoirs[label].append(item)
                else:
                    replacement = int(rng.integers(0, seen[label]))
                    if replacement < target:
                        reservoirs[label][replacement] = item
    missing = {label: count for label, count in seen.items() if count < target}
    if missing:
        raise ValueError(f"Some classes have fewer than {target} windows: {missing}")
    records = []
    for label in range(10):
        for session, start in reservoirs[label]:
            rows = session.slice(start, window_size).to_dicts()
            if label not in (0, 5):
                attack_labels = {int(row["Class"]) for row in rows if int(row["Class"]) not in (0, 5)}
                if any(value != label for value in attack_labels):
                    raise ValueError("Multiple attack types in one window; paper assumes one attack type")
            records.append({
                "session_id": session["session_id"][0],
                "start_row_id": int(rows[0]["row_id"]),
                "label": int(label),
                "frames": [_frame_string(row) for row in rows],
            })
    rng.shuffle(records)
    return records


def _split_balanced_windows(records, seed):
    """Split each balanced class independently to preserve class balance."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for label in range(10):
        group = [record for record in records if record["label"] == label]
        rng.shuffle(group)
        n_train = int(len(group) * 0.60)
        n_val = int(len(group) * 0.20)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])
    for records_for_split in splits.values():
        rng.shuffle(records_for_split)
    return splits


def feature(params):
    dataset = params["dataset"]
    paths = dataset["sequence_window"]
    processed_dir = Path(paths["processed_dir"])
    output_dir = Path(paths["featured_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # Windowing, balancing, and splitting are performed in split() so the
    # stage ordering matches the paper. This stage materializes the common
    # featured-path contract consumed by train/eval scripts.
    for split_name in ("train", "val", "test"):
        source_path = processed_dir / f"{split_name}.parquet"
        target_path = output_dir / f"{split_name}.parquet"
        pl.read_parquet(source_path).write_parquet(target_path, compression="zstd")
