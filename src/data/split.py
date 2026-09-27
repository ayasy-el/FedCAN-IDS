"""Split prepared frames or compact window references."""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from data.task import label_schema
from utils.params import load_params


def _order_columns(df):
    return ["session_id", "Timestamp"] if "Timestamp" in df.columns else ["session_id", "row_id"]


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
        name: df.filter(pl.col("session_id").is_in(sessions)).sort(_order_columns(df))
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


def _sink_lazy_frame(query, output):
    """Write one lazy result atomically without collecting it first."""
    output = Path(output)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        query.sink_parquet(
            temporary_path,
            compression="zstd",
            row_group_size=100_000,
            maintain_order=True,
            engine="streaming",
        )
        os.replace(temporary_path, output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _count_parquet_rows(path):
    return int(
        pl.scan_parquet(path)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )


def _road_stratified_frame_split(prepared_path, output_dir, seed):
    """Split ROAD frames exactly by class using bounded Arrow batches."""
    parquet = pq.ParquetFile(prepared_path)
    batch_size = 100_000
    batch_total = (parquet.metadata.num_rows + batch_size - 1) // batch_size
    counts = {}
    for batch in tqdm(
        parquet.iter_batches(columns=["Class"], batch_size=batch_size),
        total=batch_total,
        desc="Counting frame classes",
        unit="batch",
    ):
        labels = np.asarray(batch.column(0).to_numpy(zero_copy_only=False))
        values, frequencies = np.unique(labels, return_counts=True)
        for label, frequency in zip(values, frequencies):
            counts[int(label)] = counts.get(int(label), 0) + int(frequency)
    if not counts:
        raise ValueError("Prepared ROAD parquet contains no frames")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {}
    for name in ("train", "val", "test"):
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{name}.", suffix=".tmp", dir=output_dir, delete=False
        )
        temporary_paths[name] = Path(temporary.name)
        temporary.close()
    writers = {}
    output_counts = {name: 0 for name in temporary_paths}
    seen = {label: 0 for label in counts}
    source_schema = parquet.schema_arrow
    try:
        for batch in tqdm(
            parquet.iter_batches(batch_size=batch_size),
            total=batch_total,
            desc="Writing frame splits",
            unit="batch",
        ):
            frame = pl.from_arrow(batch)
            labels = np.asarray(frame["Class"].to_numpy())
            split_masks = {
                "train": np.zeros(len(frame), dtype=bool),
                "val": np.zeros(len(frame), dtype=bool),
                "test": np.zeros(len(frame), dtype=bool),
            }
            for label in counts:
                label_mask = labels == label
                count = int(label_mask.sum())
                if not count:
                    continue
                label_positions = np.arange(
                    seen[label], seen[label] + count, dtype=np.int64
                )
                seen[label] += count
                rank = _permuted_rank(
                    label_positions, counts[label], seed, label, stream=2
                )
                n_train = int(counts[label] * 0.60)
                n_val = int(counts[label] * 0.20)
                split_masks["train"][label_mask] = rank < n_train
                split_masks["val"][label_mask] = (
                    (rank >= n_train) & (rank < n_train + n_val)
                )
                split_masks["test"][label_mask] = rank >= n_train + n_val

            for name, mask in split_masks.items():
                if not mask.any():
                    continue
                table = frame.filter(pl.Series(mask)).to_arrow()
                if name not in writers:
                    writers[name] = pq.ParquetWriter(
                        temporary_paths[name],
                        table.schema,
                        compression="zstd",
                    )
                writers[name].write_table(table)
                output_counts[name] += table.num_rows

        for name in temporary_paths:
            if name not in writers:
                writers[name] = pq.ParquetWriter(
                    temporary_paths[name], source_schema, compression="zstd"
                )
            writers[name].close()
            os.replace(temporary_paths[name], output_dir / f"{name}.parquet")
    except Exception:
        for writer in writers.values():
            writer.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    if seen != counts:
        raise RuntimeError(f"ROAD class accounting mismatch: seen={seen}, counts={counts}")
    for name, count in output_counts.items():
        print(f"{name}: {count:,} frames")
    return output_counts


def _road_frame_split(prepared_path, output_dir, strategy, split, seed):
    """Stream ROAD frame splits without materializing the prepared dataset."""
    source = pl.scan_parquet(prepared_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "session_holdout":
        sessions = (
            source.select("session_id")
            .unique()
            .collect(engine="streaming")["session_id"]
            .to_list()
        )
        all_sessions = set(sessions)
        val_sessions = set(split.get("val_sessions", []))
        test_sessions = set(split.get("test_sessions", []))
        unknown = (val_sessions | test_sessions) - all_sessions
        if unknown:
            raise ValueError(f"Configured sessions do not exist: {sorted(unknown)}")
        overlap = val_sessions & test_sessions
        if overlap:
            raise ValueError(f"Validation/test sessions overlap: {sorted(overlap)}")
        assignments = {
            "train": all_sessions - val_sessions - test_sessions,
            "val": val_sessions,
            "test": test_sessions,
        }
        queries = {
            name: source.filter(pl.col("session_id").is_in(sorted(names)))
            for name, names in assignments.items()
        }
    elif strategy == "stratified_random":
        return _road_stratified_frame_split(prepared_path, output_dir, seed)
    else:
        raise ValueError(f"Unknown frame split strategy: {strategy}")

    counts = {}
    for name, query in queries.items():
        _sink_lazy_frame(query, output_dir / f"{name}.parquet")
        counts[name] = _count_parquet_rows(output_dir / f"{name}.parquet")
        print(f"{name}: {counts[name]:,} frames")
    return counts


def _candidate_labels(labels, window_size, stride, benign_labels):
    """Label windows by the first attack frame, or benign otherwise."""
    labels = np.asarray(labels, dtype=np.int16)
    starts = np.arange(0, len(labels) - window_size + 1, stride, dtype=np.int64)
    if not len(starts):
        return starts, np.empty(0, dtype=np.int16)
    attack_positions = np.flatnonzero(~np.isin(labels, tuple(benign_labels)))
    next_attack = np.full(len(labels), len(labels), dtype=np.int64)
    next_attack[attack_positions] = attack_positions
    next_attack = np.minimum.accumulate(next_attack[::-1])[::-1]
    first_attack = next_attack[starts]
    has_attack = first_attack < starts + window_size
    window_labels = labels[starts].copy()
    window_labels[has_attack] = labels[first_attack[has_attack]]
    return starts, window_labels


def _sample_window_refs(
    df, window_size, stride, downsample, seed, benign_labels, downsample_size=None
):
    """Sample compact ``(session index, start)`` references."""
    rng = np.random.default_rng(seed)
    sessions = df.sort(_order_columns(df)).partition_by(
        "session_id", maintain_order=True
    )
    counts = {}
    for session in tqdm(sessions, desc="Counting window candidates", unit="session"):
        _, candidate_labels = _candidate_labels(
            session["Class"].to_numpy(), window_size, stride, benign_labels
        )
        for label in candidate_labels:
            label = int(label)
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        raise ValueError("No candidate windows were found")

    if downsample_size is not None:
        target = int(downsample_size)
        available = min(counts.values())
        if target > available:
            raise ValueError(
                f"Requested downsample size {target:,} exceeds the smallest class "
                f"candidate count {available:,}: {counts}"
            )
    else:
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
            session["Class"].to_numpy(), window_size, stride, benign_labels
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


def _iter_prepared_sessions(prepared_path, batch_size=100_000):
    """Yield one ROAD session at a time from a Parquet batch iterator."""
    parquet = pq.ParquetFile(prepared_path)
    columns = ["session_id", "row_id", "Class"]
    available = set(parquet.schema_arrow.names)
    missing = set(columns) - available
    if missing:
        raise ValueError(f"Prepared ROAD parquet is missing columns: {sorted(missing)}")

    current_id = None
    label_parts = []
    row_id_parts = []
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        session_ids = batch.column(0).to_pylist()
        row_ids = np.asarray(batch.column(1).to_numpy(zero_copy_only=False))
        labels = np.asarray(batch.column(2).to_numpy(zero_copy_only=False), dtype=np.int16)
        boundaries = np.flatnonzero(
            np.asarray(session_ids[1:], dtype=object)
            != np.asarray(session_ids[:-1], dtype=object)
        ) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(session_ids)]
        for start, end in zip(starts, ends):
            session_id = str(session_ids[start])
            if current_id is not None and session_id != current_id:
                yield current_id, np.concatenate(row_id_parts), np.concatenate(label_parts)
                row_id_parts = []
                label_parts = []
            current_id = session_id
            row_id_parts.append(row_ids[start:end])
            label_parts.append(labels[start:end])
    if current_id is not None:
        yield current_id, np.concatenate(row_id_parts), np.concatenate(label_parts)


def _permutation_params(size, seed, label, stream):
    if size <= 1:
        return 1, 0
    candidate = abs(int(seed)) + (label + 1) * 1_000_003 + (stream + 1) * 97_003
    multiplier = candidate % size or 1
    while np.gcd(multiplier, size) != 1:
        multiplier = multiplier % size + 1
    offset = (candidate * 2 + label + stream) % size
    return multiplier, offset


def _permuted_rank(positions, size, seed, label, stream):
    multiplier, offset = _permutation_params(size, seed, label, stream)
    return (multiplier * positions + offset) % size


def _compact_rows(session_id, row_ids, window_size, label):
    row_ids = [int(value) for value in row_ids]
    session_ids = [session_id] * len(row_ids)
    return pl.DataFrame(
        {
            "window_id": [f"{session_id}:{row_id}" for row_id in row_ids],
            "session_id": session_ids,
            "start_row_id": row_ids,
            "window_size": [window_size] * len(row_ids),
            "label": [int(label)] * len(row_ids),
        }
    ).to_arrow()


def _write_compact_rows(writers, name, session_id, row_ids, window_size, label):
    for start in range(0, len(row_ids), writers.batch_size):
        batch = row_ids[start:start + writers.batch_size]
        if len(batch):
            writers.write(name, _compact_rows(session_id, batch, window_size, label))


def _road_window_split(
    prepared_path,
    output_dir,
    window_size,
    stride,
    downsample,
    seed,
    benign_labels,
    downsample_size=None,
):
    """Create ROAD window references with bounded per-session memory."""
    counts = {}
    for _, _, labels in tqdm(
        _iter_prepared_sessions(prepared_path),
        desc="Counting window candidates",
        unit="session",
    ):
        _, candidate_labels = _candidate_labels(labels, window_size, stride, benign_labels)
        if len(candidate_labels):
            values, frequencies = np.unique(candidate_labels, return_counts=True)
            for label, frequency in zip(values, frequencies):
                counts[int(label)] = counts.get(int(label), 0) + int(frequency)
    if not counts:
        raise ValueError("No candidate windows were found")

    if downsample_size is not None:
        target = int(downsample_size)
        available = min(counts.values())
        if target > available:
            raise ValueError(
                f"Requested downsample size {target:,} exceeds the smallest class "
                f"candidate count {available:,}: {counts}"
            )
    else:
        target = min(counts.values()) if downsample else None
    if target is not None:
        print(f"Automatic downsample target: {target:,} windows per class")

    output_dir.mkdir(parents=True, exist_ok=True)
    writers = _CompactParquetWriters(output_dir)
    seen = {label: 0 for label in counts}
    try:
        for session_id, row_ids, labels in tqdm(
            _iter_prepared_sessions(prepared_path),
            desc="Writing window references",
            unit="session",
        ):
            starts, candidate_labels = _candidate_labels(labels, window_size, stride, benign_labels)
            for label in counts:
                label_mask = candidate_labels == label
                label_starts = starts[label_mask]
                if not len(label_starts):
                    continue
                positions = np.arange(
                    seen[label], seen[label] + len(label_starts), dtype=np.int64
                )
                seen[label] += len(label_starts)

                if target is not None:
                    selection_rank = _permuted_rank(
                        positions, counts[label], seed, label, stream=0
                    )
                    selected = selection_rank < target
                    label_starts = label_starts[selected]
                    positions = selection_rank[selected]
                    split_size = target
                else:
                    split_size = counts[label]
                if not len(label_starts):
                    continue

                split_rank = _permuted_rank(
                    positions, split_size, seed, label, stream=1
                )
                window_row_ids = row_ids[label_starts]
                n_train = int(split_size * 0.60)
                n_val = int(split_size * 0.20)
                partitions = {
                    "train": split_rank < n_train,
                    "val": (split_rank >= n_train) & (split_rank < n_train + n_val),
                    "test": split_rank >= n_train + n_val,
                }
                for name, mask in partitions.items():
                    _write_compact_rows(
                        writers,
                        name,
                        session_id,
                        window_row_ids[mask],
                        window_size,
                        label,
                    )
    finally:
        writers.close()
    for name, count in writers.counts.items():
        print(f"{name}: {count:,} compact window references")
    return writers.counts


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


def _window_split(
    prepared_path,
    output_dir,
    window_size,
    stride,
    downsample,
    seed,
    benign_labels,
    downsample_size=None,
):
    source = pl.read_parquet(prepared_path)
    sessions, references = _sample_window_refs(
        source, window_size, stride, downsample, seed, benign_labels, downsample_size
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


def run(params, downsample_size=None):
    dataset = params["dataset"]
    split = params["split"]
    output_dir = Path(dataset["processed_dir"])
    unit = split.get("unit", "frame")
    strategy = split.get("strategy", "session_holdout")
    seed = int(split.get("random_seed", 42))
    benign_labels = label_schema(
        params.get("prepare", {}).get("label_schema", "five_class"),
        dataset.get("variant"),
    )["benign_labels"]
    road_adapter = dataset.get("ingest_adapter") == "road_raw"

    if unit == "window":
        if strategy != "stratified_random":
            raise ValueError("Window unit currently supports strategy='stratified_random' only")
        if road_adapter:
            _road_window_split(
                dataset["prepared_path"],
                output_dir,
                int(split["window_size"]),
                int(split["stride"]),
                bool(split.get("downsample", False)),
                seed,
                benign_labels,
                downsample_size,
            )
            return
        _window_split(
            dataset["prepared_path"],
            output_dir,
            int(split["window_size"]),
            int(split["stride"]),
            bool(split.get("downsample", False)),
            seed,
            benign_labels,
            downsample_size,
        )
        return

    if road_adapter:
        if unit != "frame":
            raise ValueError("split.unit must be either 'frame' or 'window'")
        _road_frame_split(
            dataset["prepared_path"], output_dir, strategy, split, seed
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
    parser = argparse.ArgumentParser(description="Split prepared frames or window references")
    parser.add_argument(
        "--downsample-size",
        type=int,
        default=None,
        help="Optional exact number of windows to retain per class",
    )
    args = parser.parse_args()
    if args.downsample_size is not None and args.downsample_size <= 0:
        parser.error("--downsample-size must be a positive integer")
    run(load_params(), downsample_size=args.downsample_size)
