"""Single shared ingestion stage for every model."""

from pathlib import Path

import polars as pl

from utils.params import load_params


ATTACK_COLUMNS = ("SubClass", "Class")


def run(params):
    dataset = params["dataset"]
    raw_dir = Path(dataset["raw_dir"])
    configured_files = dataset.get("raw_files")
    if configured_files:
        files = []
        for name in configured_files:
            direct = raw_dir / name
            matches = [direct] if direct.exists() else sorted(raw_dir.rglob(name))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one raw file named {name!r} under {raw_dir}, "
                    f"found {len(matches)}"
                )
            files.append(matches[0])
    else:
        files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Configured raw files do not exist: {missing}")

    frames = []
    for path in files:
        df = pl.read_csv(path, infer_schema_length=1000)
        source_label = next((column for column in ATTACK_COLUMNS if column in df.columns), None)
        if source_label is None:
            df = df.with_columns(pl.lit("Normal").alias("attack_type"))
        else:
            df = df.with_columns(
                pl.col(source_label).fill_null("Normal").cast(pl.String).alias("attack_type")
            )
        frames.append(df.with_columns(pl.lit(path.stem).alias("session_id")))

    result = pl.concat(frames, how="diagonal")
    result = (
        result.with_columns(pl.col("DLC").cast(pl.UInt8))
        .with_columns(
            pl.col("Data")
            .str.split(" ")
            .list.to_struct(fields=[f"Data_{i}" for i in range(8)])
            .alias("Bytes")
        )
        .unnest("Bytes")
        .with_columns(*[pl.col(f"Data_{i}").fill_null("PAD") for i in range(8)])
        .drop("Class", "SubClass", "Data", strict=False)
        .select("session_id", "Timestamp", "Arbitration_ID", "DLC", *[f"Data_{i}" for i in range(8)], "attack_type")
        .with_row_index("row_id")
    )
    output = Path(dataset["interim_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd")
    print(f"Ingested {len(files)} raw files and {len(result):,} frames")


if __name__ == "__main__":
    run(load_params())
