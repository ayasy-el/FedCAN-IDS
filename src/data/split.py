"""Split prepared frames or compact window references."""

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


def _candidate_labels(labels, window_size, stride):
    """Label windows by the first attack frame, or benign otherwise."""
    labels = np.asarray(labels, dtype=np.int16)
    starts = np.arange(0, len(labels) - window_size + 1, stride, dtype=np.int64)
    if not len(starts):
        return starts, np.empty(0, dtype=np.int16)
    attack_positions = np.flatnonzero(~np.isin(labels, (0, 5)))
    next_attack = np.full(len(labels), len(labels), dtype=np.int64)
    next_attack[attack_positions] = attack_positions
    next_attack = np.minimum.accumulate(next_attack[::-1])[::-1]
    first_attack = next_attack[starts]
    has_attack = first_attack < starts + window_size
    window_labels = labels[starts].copy()
    window_labels[has_attack] = labels[first_attack[has_attack]]
    return starts, window_labels


def _sample_window_refs(df, window_size, stride, downsample, seed):
    """Sample compact ``(session index, start)`` references."""
    rng = np.random.default_rng(seed)
    sessions = df.sort(["session_id", "Timestamp"]).partition_by(
        "session_id", maintain_order=True
    )
    counts = {}
    for session in tqdm(sessions, desc="Counting window candidates", unit="session"):
        _, candidate_labels = _candidate_labels(
            session["Class"].to_numpy(), window_size, stride
        )
        for label in candidate_labels:
            label = int(label)
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise ValueError("No candidate windows were found")

    target = min(counts.values()) if downsample else None
    if target is not None:
        print(f"Automatic downsample target: {target} windows per class")

    sample_rng = np.random.default_rng(seed)
    selected_positions = {
        label: np.sort(
            sample_rng.choice(count, size=target, replace=False)
            if target is not None
            else np.arange(count, dtype=np.int64)
        )
        for label, count in counts.items()
    }
    references = {
        label: np.empty((len(positions), 2), dtype=np.int64)
        for label, positions in selected_positions.items()
    }
    cursors = {label: 0 for label in counts}
    seen = {label: 0 for label in counts}
    for session_index, session in enumerate(
        tqdm(sessions, desc="Sampling window references", unit="session")
    ):
        starts, candidate_labels = _candidate_labels(
            session["Class"].to_numpy(), window_size, stride
        )
        for start, candidate_label in zip(starts, candidate_labels):
            label = int(candidate_label)
            position = seen[label]
            seen[label] += 1
            cursor = cursors[label]
            positions = selected_positions[label]
            if cursor < len(positions) and position == positions[cursor]:
                references[label][cursor] = (session_index, int(start))
                cursors[label] += 1
    return sessions, references


class _CompactParquetWriters:
    def __init__(self, output_dir, batch_size=10_000):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.writers = {}
        self.counts = {name: 0 for name in ("train", "val", "test")}
        self.empty_table = None

    def write(self, name, table):
        if self.empty_table is None:
            self.empty_table = table.slice(0, 0)
        if name not in self.writers:
            self.writers[name] = pq.ParquetWriter(
                self.output_dir / f"{name}.parquet",
                table.schema,
                compression="zstd",
            )
        self.writers[name].write_table(table)
        self.counts[name] += table.num_rows

    def close(self):
        if self.empty_table is not None:
            for name in self.counts:
                if name not in self.writers:
                    writer = pq.ParquetWriter(
                        self.output_dir / f"{name}.parquet",
                        self.empty_table.schema,
                        compression="zstd",
                    )
                    writer.write_table(self.empty_table)
                    writer.close()
        for writer in self.writers.values():
            writer.close()


def _compact_table(sessions, references, window_size, label):
    session_indices = references[:, 0]
    starts = references[:, 1]
    session_ids = [str(sessions[int(index)]["session_id"][0]) for index in session_indices]
    row_ids = [int(sessions[int(index)]["row_id"][int(start)]) for index, start in references]
    return pl.DataFrame({
        "window_id": [f"{session}:{row_id}" for session, row_id in zip(session_ids, row_ids)],
        "session_id": session_ids,
        "start_row_id": row_ids,
        "window_size": [window_size] * len(references),
        "label": [int(label)] * len(references),
    }).to_arrow()


def _window_split(prepared_path, output_dir, window_size, stride, downsample, seed):
    source = pl.read_parquet(prepared_path)
    sessions, references = _sample_window_refs(
        source, window_size, stride, downsample, seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = _CompactParquetWriters(output_dir)
    split_rng = np.random.default_rng(seed)
    try:
        for label, group in references.items():
            order = split_rng.permutation(len(group))
            n_train = int(len(group) * 0.60)
            n_val = int(len(group) * 0.20)
            partitions = {
                "train": order[:n_train],
                "val": order[n_train:n_train + n_val],
                "test": order[n_train + n_val:],
            }
            for name, indices in partitions.items():
                for start in range(0, len(indices), writers.batch_size):
                    batch = group[indices[start:start + writers.batch_size]]
                    if len(batch):
                        writers.write(name, _compact_table(sessions, batch, window_size, label))
    finally:
        writers.close()
    for name, count in writers.counts.items():
        print(f"{name}: {count:,} compact window references")
    return writers.counts


def run(params):
    dataset = params["dataset"]
    split = params["split"]
    output_dir = Path(dataset["processed_dir"])
    unit = split.get("unit", "frame")
    strategy = split.get("strategy", "session_holdout")
    seed = int(split.get("random_seed", 42))

    if unit == "window":
        if strategy != "stratified_random":
            raise ValueError("Window unit currently supports strategy='stratified_random' only")
        _window_split(
            dataset["prepared_path"],
            output_dir,
            int(split["window_size"]),
            int(split["stride"]),
            bool(split.get("downsample", False)),
            seed,
        )
        return

    source = pl.read_parquet(dataset["prepared_path"])
    print(f"Splitting {len(source):,} prepared frames with strategy={strategy!r}")
    if unit != "frame":
        raise ValueError("split.unit must be either 'frame' or 'window'")
    if strategy == "session_holdout":
        outputs = _session_split(source, split)
    elif strategy == "stratified_random":
        outputs = _stratified_frame_split(source, seed)
    else:
        raise ValueError(f"Unknown frame split strategy: {strategy}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in outputs.items():
        result.write_parquet(output_dir / f"{name}.parquet", compression="zstd")
        print(f"{name}: {len(result):,} frames")


if __name__ == "__main__":
    run(load_params())
