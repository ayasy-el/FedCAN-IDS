from pathlib import Path

import polars as pl

from utils.params import load_params


def add_deltatime_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["session_id", "Timestamp"])
    return df.with_columns(

        # Delta time antar frame (per session_id)
        pl.col("Timestamp")
        .diff()
        .over("session_id")
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("Deltatime"),
        
        # Delta time antar frame dengan Arbitration_ID yang sama (per session_id)
        pl.col("Timestamp")
        .diff()
        .over(["session_id", "Arbitration_ID"])
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("Delta_Id"),
    )


def build_features(input_path: Path, output_path: Path):
    print(f"Processing {input_path.name}")
    df = add_deltatime_features(pl.read_parquet(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    dataset = load_params("dataset")
    output_dir = Path(dataset["featured_dir"])
    for split in ("train", "val", "test"):
        build_features(
            Path(dataset["processed_dir"]) / f"{split}.parquet",
            output_dir / f"{split}.parquet",
        )
