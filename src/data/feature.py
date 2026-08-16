from pathlib import Path

import polars as pl

from utils.params import load_params

WINDOW_MS = load_params("feature")["window_ms"]


###############################################################################
# Helpers
###############################################################################


def payload_to_int(x):
    if x is None:
        return None

    if x == "PAD":
        return None

    if isinstance(x, str):
        return int(x, 16)

    return int(x)


def xor_bitcount(curr, prev):
    curr = payload_to_int(curr)
    prev = payload_to_int(prev)

    if curr is None or prev is None:
        return 0

    return (curr ^ prev).bit_count()


###############################################################################
# Per-ID temporal features
###############################################################################


def add_delta_t(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("Timestamp")
        .diff()
        .over(["session_id", "Arbitration_ID"])
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("delta_t")
    )


def add_delta_t_z(df: pl.DataFrame) -> pl.DataFrame:
    median = pl.col("delta_t").median().over(["session_id", "Arbitration_ID"])

    mad = (
        (pl.col("delta_t") - median)
        .abs()
        .median()
        .over(["session_id", "Arbitration_ID"])
    )

    return df.with_columns(
        ((pl.col("delta_t") - median) / (1.4826 * mad + 1e-9))
        .fill_null(0.0)
        .fill_nan(0.0)
        .cast(pl.Float32)
        .alias("delta_t_z")
    )


def add_jitter(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col("delta_t")
            - pl.col("delta_t").shift(1).over(["session_id", "Arbitration_ID"])
        )
        .abs()
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("jitter")
    )


def add_hamming_distance(
    df: pl.DataFrame,
) -> pl.DataFrame:

    exprs = []

    for i in range(8):
        c = f"Data_{i}"

        prev = pl.col(c).shift(1).over(["session_id", "Arbitration_ID"])

        expr = pl.struct(
            [
                pl.col(c).alias("curr"),
                prev.alias("prev"),
            ]
        ).map_elements(
            lambda x: xor_bitcount(
                x["curr"],
                x["prev"],
            ),
            return_dtype=pl.UInt8,
        )

        exprs.append(expr)

    return df.with_columns(
        pl.sum_horizontal(exprs).cast(pl.UInt8).alias("hamming_distance")
    )


###############################################################################
# Global window features
###############################################################################


def add_window_id(
    df: pl.DataFrame,
) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("Timestamp") * 1000 / WINDOW_MS).floor().cast(pl.Int64)).alias(
            "window_id"
        )
    )


def add_frame_rate(
    df: pl.DataFrame,
) -> pl.DataFrame:

    frame_rate = df.group_by(["session_id", "window_id"]).agg(
        pl.len().alias("frame_rate_20ms")
    )

    return df.join(
        frame_rate,
        on=["session_id", "window_id"],
        how="left",
    )


def add_id_rate(
    df: pl.DataFrame,
) -> pl.DataFrame:

    id_rate = df.group_by(
        [
            "session_id",
            "window_id",
            "Arbitration_ID",
        ]
    ).agg(pl.len().alias("id_rate_20ms"))

    return df.join(
        id_rate,
        on=[
            "session_id",
            "window_id",
            "Arbitration_ID",
        ],
        how="left",
    )


def add_entropy(
    df: pl.DataFrame,
) -> pl.DataFrame:

    counts = df.group_by(
        [
            "session_id",
            "window_id",
            "Arbitration_ID",
        ]
    ).agg(pl.len().alias("n"))

    frame_rate = df.group_by(["session_id", "window_id"]).agg(pl.len().alias("total"))

    entropy = (
        counts.join(
            frame_rate,
            on=["session_id", "window_id"],
        )
        .with_columns((pl.col("n") / pl.col("total")).alias("p"))
        .with_columns((-pl.col("p") * pl.col("p").log(base=2)).alias("h"))
        .group_by(["session_id", "window_id"])
        .agg(pl.sum("h").alias("id_entropy_20ms"))
    )

    return df.join(
        entropy,
        on=["session_id", "window_id"],
        how="left",
    )


def add_dominant_ratio(
    df: pl.DataFrame,
) -> pl.DataFrame:

    counts = df.group_by(
        [
            "session_id",
            "window_id",
            "Arbitration_ID",
        ]
    ).agg(pl.len().alias("n"))

    total = df.group_by(["session_id", "window_id"]).agg(pl.len().alias("total"))

    ratio = (
        counts.group_by(["session_id", "window_id"])
        .agg(pl.max("n").alias("max_count"))
        .join(
            total,
            on=["session_id", "window_id"],
        )
        .with_columns(
            (pl.col("max_count") / pl.col("total"))
            .cast(pl.Float32)
            .alias("dominant_id_ratio_20ms")
        )
        .select(
            [
                "session_id",
                "window_id",
                "dominant_id_ratio_20ms",
            ]
        )
    )

    return df.join(
        ratio,
        on=["session_id", "window_id"],
        how="left",
    )


def add_bus_load_proxy(
    df: pl.DataFrame,
) -> pl.DataFrame:

    p95 = df["frame_rate_20ms"].quantile(0.95)

    if p95 is None or p95 <= 0:
        p95 = 1.0

    return df.with_columns(
        (pl.col("frame_rate_20ms") / p95)
        .clip(0.0, 1.0)
        .cast(pl.Float32)
        .alias("bus_load_proxy_20ms")
    )


###############################################################################
# Pipeline
###############################################################################


def build_features(
    input_path: Path,
    output_path: Path,
):
    print(f"Processing {input_path.name}")

    df = pl.read_parquet(input_path)

    df = (
        df.pipe(add_delta_t)
        .pipe(add_delta_t_z)
        .pipe(add_jitter)
        .pipe(add_hamming_distance)
        .pipe(add_window_id)
        .pipe(add_frame_rate)
        .pipe(add_id_rate)
        .pipe(add_entropy)
        .pipe(add_dominant_ratio)
        .pipe(add_bus_load_proxy)
        .drop("window_id")
    )

    df.write_parquet(
        output_path,
        compression="zstd",
    )

    print(f"Saved: {output_path}")
    print(df.head())


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    _dataset_params = load_params("dataset")
    input_dir = Path(_dataset_params["processed_dir"])
    output_dir = Path(_dataset_params["featured_dir"])

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        build_features(
            input_dir / f"{split}.parquet",
            output_dir / f"{split}.parquet",
        )
