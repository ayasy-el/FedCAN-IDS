"""Split prepared frames or windows using one configurable data stage."""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from utils.params import load_params


def _session_split(df, split):
    val_sessions = split.get("val_sessions", [])
    test_sessions = split.get("test_sessions", [])
    all_sessions = set(df["session_id"].unique().to_list())
    unknown = (set(val_sessions) | set(test_sessions)) - all_sessions
    if unknown:
        raise ValueError(f"Configured sessions do not exist: {sorted(unknown)}")
    overlap = set(val_sessions) & set(test_sessions)
    if overlap:
        raise ValueError(f"Validation/test sessions overlap: {sorted(overlap)}")
    assignments = {
        "train": sorted(all_sessions - set(val_sessions) - set(test_sessions)),
        "val": list(val_sessions),
        "test": list(test_sessions),
    }
    return {
        name: df.filter(pl.col("session_id").is_in(sessions)).sort(["session_id", "Timestamp"])
        for name, sessions in assignments.items()
    }


def _stratified_frame_split(df, seed):
    rng = np.random.default_rng(seed)
    outputs = {"train": [], "val": [], "test": []}
    for label in sorted(df["Class"].unique().to_list()):
        group = df.filter(pl.col("Class") == label)
        indices = np.arange(len(group))
        rng.shuffle(indices)
        n_train = int(len(indices) * 0.60)
        n_val = int(len(indices) * 0.20)
        outputs["train"].append(group[indices[:n_train]])
        outputs["val"].append(group[indices[n_train:n_train + n_val]])
        outputs["test"].append(group[indices[n_train + n_val:]])
    return {name: pl.concat(groups) if groups else df.head(0) for name, groups in outputs.items()}


def _frame_string(row):
    values = [str(row["Arbitration_ID"]).upper().removeprefix("0X"), format(int(row["DLC"]), "X")]
    values.extend(str(row[f"Data_{i}"]).upper() for i in range(8))
    return " ".join(values)


def _candidate_labels(labels, window_size, stride):
    """Label each window by its first attack frame, or benign otherwise."""
    labels = np.asarray(labels, dtype=np.int16)
    starts = np.arange(0, len(labels) - window_size + 1, stride, dtype=np.int64)
    if not len(starts):
        return starts, np.empty(0, dtype=np.int16)

    attack_positions = np.flatnonzero(~np.isin(labels, (0, 5)))
    next_attack = np.full(len(labels), len(labels), dtype=np.int64)
    next_attack[attack_positions] = attack_positions
    next_attack = np.minimum.accumulate(next_attack[::-1])[::-1]
    first_attack = next_attack[starts]
    has_attack = first_attack < (starts + window_size)
    window_labels = labels[starts].copy()
    window_labels[has_attack] = labels[first_attack[has_attack]]
    return starts, window_labels


def _window_label(rows):
    """Use an attack label when any frame in the window is an attack."""
    for row in rows:
        label = int(row["Class"])
        if label not in (0, 5):
            return label
    return int(rows[0]["Class"])


def _materialize_window_record(session, start, window_size):
    """Build the wide window row only after its start was sampled."""
    rows = session.slice(start, window_size).to_dicts()
    first = rows[0]
    return {
        "window_id": f"{first['session_id']}:{first['row_id']}",
        "session_id": first["session_id"],
        "start_row_id": int(first["row_id"]),
        "label": _window_label(rows),
        "frame_labels": [int(row["Class"]) for row in rows],
        "frames": [_frame_string(row) for row in rows],
        "Arbitration_ID": [str(row["Arbitration_ID"]) for row in rows],
        "DLC": [int(row["DLC"]) for row in rows],
        **{f"Data_{i}": [str(row[f"Data_{i}"]) for row in rows] for i in range(8)},
        "Delta_Id": [float(row["Delta_Id"]) for row in rows],
        "Deltatime": [float(row["Deltatime"]) for row in rows],
    }


def _sample_window_refs(df, window_size, stride, downsample, seed):
    """Sample lightweight window references before materializing window data."""
    rng = np.random.default_rng(seed)
    sessions = df.sort(["session_id", "Timestamp"]).partition_by(
        "session_id", maintain_order=True
    )
    counts = {}
    for session in tqdm(sessions, desc="Counting window candidates", unit="session"):
        labels = session["Class"].to_numpy()
        if len(labels) < window_size:
            continue
        _, candidate_labels = _candidate_labels(labels, window_size, stride)
        for label in candidate_labels:
            label = int(label)
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise ValueError("No candidate windows were found")

    for label, count in sorted(counts.items()):
        print(f"class {label}: {count:,}")

    target = min(counts.values()) if downsample else None
    if target is not None:
        print(f"Automatic downsample target: {target} windows per class")

    reservoirs = {label: [] for label in counts}
    seen = {label: 0 for label in counts}
    for session_index, session in enumerate(
        tqdm(sessions, desc="Sampling window references", unit="session")
    ):
        labels = session["Class"].to_numpy()
        starts, candidate_labels = _candidate_labels(labels, window_size, stride)
        for start, candidate_label in zip(starts, candidate_labels):
            label = int(candidate_label)
            seen[label] += 1
            reference = (session_index, start)
            if target is None or len(reservoirs[label]) < target:
                reservoirs[label].append(reference)
            else:
                replacement = int(rng.integers(0, seen[label]))
                if replacement < target:
                    reservoirs[label][replacement] = reference
    return sessions, reservoirs


def _split_window_refs(references, rng):
    """Create shuffled train/val/test references without materializing rows."""
    outputs = {"train": [], "val": [], "test": []}
    for label, group in references.items():
        group = list(group)
        rng.shuffle(group)
        tagged = [(label, session_index, start) for session_index, start in group]
        n_train = int(len(tagged) * 0.60)
        n_val = int(len(tagged) * 0.20)
        outputs["train"].extend(tagged[:n_train])
        outputs["val"].extend(tagged[n_train:n_train + n_val])
        outputs["test"].extend(tagged[n_train + n_val:])
    for group in outputs.values():
        rng.shuffle(group)
    return outputs


class _WindowParquetWriters:
    """Write window records in bounded batches using Arrow Parquet writers."""

    def __init__(self, output_dir, batch_size=64):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.buffers = {name: [] for name in ("train", "val", "test")}
        self.writers = {}
        self.counts = {name: 0 for name in self.buffers}
        self.empty_table = None

    def append(self, split_name, record):
        buffer = self.buffers[split_name]
        buffer.append(record)
        if len(buffer) >= self.batch_size:
            self._flush(split_name)

    def _flush(self, split_name):
        buffer = self.buffers[split_name]
        if not buffer:
            return
        table = pl.DataFrame(buffer).to_arrow()
        if self.empty_table is None:
            self.empty_table = table.slice(0, 0)
        writer = self.writers.get(split_name)
        if writer is None:
            writer = pq.ParquetWriter(
                self.output_dir / f"{split_name}.parquet",
                table.schema,
                compression="zstd",
            )
            self.writers[split_name] = writer
        writer.write_table(table)
        self.counts[split_name] += len(buffer)
        buffer.clear()

    def close(self):
        for split_name in self.buffers:
            self._flush(split_name)
        if self.empty_table is not None:
            for split_name in self.buffers:
                if split_name not in self.writers:
                    writer = pq.ParquetWriter(
                        self.output_dir / f"{split_name}.parquet",
                        self.empty_table.schema,
                        compression="zstd",
                    )
                    writer.write_table(self.empty_table)
                    writer.close()
        for writer in self.writers.values():
            writer.close()


def _write_window_splits(sessions, references, window_size, output_dir):
    writers = _WindowParquetWriters(output_dir)
    total = sum(len(group) for group in references.values())
    try:
        with tqdm(total=total, desc="Materializing sampled windows", unit="window") as progress:
            for split_name, group in references.items():
                for _, session_index, start in group:
                    writers.append(
                        split_name,
                        _materialize_window_record(
                            sessions[session_index], start, window_size
                        ),
                    )
                    progress.update(1)
    finally:
        writers.close()
    return writers.counts


def _stratified_window_split(prepared_path, window_size, stride, downsample, seed, output_dir):
    # Read only inside this function so the original frame table can be
    # released immediately after session partitioning.
    df = pl.read_parquet(prepared_path)
    rng = np.random.default_rng(seed)
    sessions, references = _sample_window_refs(
        df, window_size, stride, downsample, seed
    )
    del df
    split_references = _split_window_refs(references, rng)
    return _write_window_splits(sessions, split_references, window_size, output_dir)


def run(params):
    dataset = params["dataset"]
    split = params["split"]
    output_dir = Path(dataset["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    unit = split.get("unit", "frame")
    strategy = split.get("strategy", "session_holdout")
    seed = int(split.get("random_seed", 42))
    if unit == "window":
        if strategy != "stratified_random":
            raise ValueError("Window unit currently supports strategy='stratified_random' only")
        print(f"Splitting prepared frames into windows with strategy={strategy!r}")
        counts = _stratified_window_split(
            dataset["prepared_path"],
            int(split["window_size"]),
            int(split["stride"]),
            bool(split.get("downsample", False)),
            seed,
            output_dir,
        )
        for name, count in counts.items():
            print(f"{name}: {count:,} {unit}s -> {output_dir / f'{name}.parquet'}")
        return

    source = pl.read_parquet(dataset["prepared_path"])
    print(f"Splitting {len(source):,} prepared {unit}s with strategy={strategy!r}")

    if unit == "frame":
        if strategy == "session_holdout":
            outputs = _session_split(source, split)
        elif strategy == "stratified_random":
            outputs = _stratified_frame_split(source, seed)
        else:
            raise ValueError(f"Unknown frame split strategy: {strategy}")
    else:
        raise ValueError("split.unit must be either 'frame' or 'window'")

    for name, result in outputs.items():
        path = output_dir / f"{name}.parquet"
        if isinstance(result, pl.DataFrame):
            result.write_parquet(path, compression="zstd")
        else:
            pl.DataFrame(result).write_parquet(path, compression="zstd")
        print(f"{name}: {len(result):,} {unit}s -> {path}")


if __name__ == "__main__":
    run(load_params())
