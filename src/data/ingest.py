import polars as pl
from pathlib import Path

subclass_map = {
    "Normal": 0,
    "Flooding": 1,
    "Fuzzing": 2,
    "Spoofing": 3,
    "Replay": 4,
}

df = (
    pl.concat(
        (
            # session_id = nama file asal (mis. "Pre_train_D_1").
            # WAJIB ada sebelum split & feature engineering karena Timestamp
            # antar file TIDAK sinkron (sebagian relatif ke awal capture,
            # sebagian epoch absolut, dan beda sesi capture beda pula epoch-nya).
            # Semua operasi urutan-sensitif (sort, diff, shift, window) di
            # tahap berikutnya harus dipartisi per session_id, tidak pernah
            # dibandingkan lintas sesi.
            pl.scan_csv(f).with_columns(
                pl.lit(f.stem).alias("session_id")
            )
            for f in Path("data/raw/Car_Hacking_Challenge_Dataset").rglob("*.csv")
        ),
        how="diagonal",
    )

    # DLC to uint8
    .with_columns(
        pl.col("DLC")
        .cast(pl.UInt8)
        .alias("DLC")
    )

    # Fill missing subclass for normal samples
    .with_columns(
        pl.when(
            (pl.col("SubClass").is_null()) &
            (pl.col("Class") == "Normal")
        )
        .then(pl.lit("Normal"))
        .otherwise(pl.col("SubClass"))
        .alias("SubClass")
    )

    # Encode label
    .drop("Class")
    .with_columns(
        pl.col("SubClass")
        .replace_strict(subclass_map)
        .cast(pl.UInt8)
        .alias("Class")
    )
    .drop("SubClass")

    # Split payload menjadi 8 byte
    .with_columns(
        pl.col("Data")
        .str.split(" ")
        .list.to_struct(
            fields=[f"Data_{i}" for i in range(8)],
        )
        .alias("Bytes")
    )
    .unnest("Bytes")
    .drop("Data")

    # Padding
    .with_columns(
        *[
            pl.col(f"Data_{i}")
            .fill_null("PAD")
            .alias(f"Data_{i}")
            for i in range(8)
        ]
    )

    # Organize the column order.
    .select(
        "session_id",
        "Timestamp",
        "Arbitration_ID",
        "DLC",
        *[f"Data_{i}" for i in range(8)],
        "Class",
    )

    # row_id unik global, berguna untuk unit test anti-leakage
    # (memastikan tidak ada baris yang sama muncul di lebih dari satu split)
    .with_row_index("row_id")
)

print(df.collect().head())

df.sink_parquet(
    "data/interim/car_hacking_with_session.parquet",
    compression="zstd",
)