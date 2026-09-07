from pathlib import Path

import polars as pl

from utils.params import load_params

_split_params = load_params("split")

VAL_SESSIONS = _split_params["val_sessions"]
TEST_SESSIONS = _split_params["test_sessions"]
# Training uses all sessions not explicitly assigned to validation or test.

_dataset_params = load_params("dataset")

input_path = Path(_dataset_params["interim_path"])
output_dir = Path(_dataset_params["processed_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(input_path)

all_sessions = set(df["session_id"].unique().to_list())


# Validate that all configured sessions exist in the dataset.
for name, sessions in [
    ("VAL_SESSIONS", VAL_SESSIONS),
    ("TEST_SESSIONS", TEST_SESSIONS),
]:
    unknown = set(sessions) - all_sessions
    if unknown:
        raise RuntimeError(
            f"{name} references session_id values not present in the dataset: "
            f"{sorted(unknown)}\n"
            f"Available session_id values: {sorted(all_sessions)}"
        )

# Prevent data leakage between validation and test splits.
overlap = set(VAL_SESSIONS) & set(TEST_SESSIONS)
if overlap:
    raise RuntimeError(
        f"VAL_SESSIONS and TEST_SESSIONS must not overlap: {sorted(overlap)}"
    )

train_sessions = sorted(all_sessions - set(VAL_SESSIONS) - set(TEST_SESSIONS))

sessions_by_split = {
    "train": train_sessions,
    "val": VAL_SESSIONS,
    "test": TEST_SESSIONS,
}

expected_classes = set(df["Class"].unique().to_list())
total_rows = len(df)


# Collect per-session statistics for reproducibility and sanity checks.
session_stats = (
    df.group_by("session_id")
    .agg(
        pl.len().alias("n_rows"),
        pl.col("Class").unique().alias("classes"),
    )
    .sort("session_id")
    .to_dicts()
)

print(f"Number of sessions: {len(session_stats)}")
for s in session_stats:
    print(
        f"  {s['session_id']:30} "
        f"n_rows={s['n_rows']:>10,}  "
        f"classes={sorted(s['classes'])}"
    )

print("\nSession split (fixed configuration):")
for name in ["train", "val", "test"]:
    print(f"  {name:5}: {sessions_by_split[name]}")


def select_sessions(session_ids: list[str]) -> pl.DataFrame:
    return df.filter(pl.col("session_id").is_in(session_ids)).sort(
        ["session_id", "Timestamp"]
    )


train_df = select_sessions(sessions_by_split["train"])
val_df = select_sessions(sessions_by_split["val"])
test_df = select_sessions(sessions_by_split["test"])

splits = {"train": train_df, "val": val_df, "test": test_df}

print("\nRow proportions:")
for name, split_df in splits.items():
    count = len(split_df)
    print(f"  {name:5}: {count:,} ({count / total_rows:.2%})")

print("\nClass distribution by split:")
any_missing = False

for name, split_df in splits.items():
    actual_classes = set(split_df["Class"].unique().to_list())
    missing_classes = expected_classes - actual_classes

    print(f"\n-- {name} --")

    class_distribution = (
        split_df.group_by("Class")
        .len()
        .rename({"len": "count"})
        .with_columns(
            (pl.col("count") / pl.col("count").sum() * 100).alias("percentage")
        )
        .with_columns(
            pl.col("percentage").map_elements(
                lambda x: f"{x:.2f}%",
                return_dtype=pl.String,
            )
        )
        .sort("Class")
    )

    print(class_distribution)

    if missing_classes:
        any_missing = True
        print(
            f"  !! WARNING: class {sorted(missing_classes)} "
            f"is missing in split '{name}'"
        )

if any_missing:
    print(
        "\nWARNING: At least one split is missing one or more classes. "
        "Consider reassigning validation/test sessions to improve class "
        "coverage, or explicitly document this limitation in the report."
    )
else:
    print("\nOK: all classes are represented in train, validation, and test.")


train_df.write_parquet(output_dir / "train.parquet", compression="zstd")
val_df.write_parquet(output_dir / "val.parquet", compression="zstd")
test_df.write_parquet(output_dir / "test.parquet", compression="zstd")
