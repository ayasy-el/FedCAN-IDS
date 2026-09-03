from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf


ID_MAX = 2047.0  # CAN ID standar 11-bit: nilai valid 0..2047
DLC_MAX = 8.0
BYTE_MAX = 255.0
PAD_SENTINEL = (
    -1.0
)  # nilai valid selalu >=0 setelah normalisasi -> -1 aman jadi penanda PAD


class CANDataset:
    def __init__(
        self,
        parquet_path,
        batch_size=256,
        shuffle=True,
    ):
        self.parquet_path = Path(parquet_path)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.df = pl.read_parquet(self.parquet_path)

        self.prepare()

    def prepare(self):
        df = self.df

        #
        # Numeric Encoding
        #
        # Arbitration_ID "4F1" -> 1265 -> 1265/2047 (dinormalisasi [0,1])
        # DLC -> DLC/8 (dinormalisasi [0,1])
        # Data_i "FF" -> 255/255=1.0 ; "PAD" -> null -> PAD_SENTINEL (-1.0)
        #
        # Menggantikan tokenisasi diskrit lama (tokens/token_types) --
        # lihat NumericFeatureProjection di src/model/embeddings.py.
        #
        df = df.with_columns(
            (
                pl.col("Arbitration_ID")
                .str.to_integer(base=16, strict=False)
                .fill_null(0)
                .cast(pl.Float32)
                / ID_MAX
            ).alias("id_norm"),
            (pl.col("DLC").cast(pl.Float32) / DLC_MAX).alias("dlc_norm"),
        )

        payload_exprs = []

        for i in range(8):
            c = f"Data_{i}"

            payload_exprs.append(
                (
                    pl.col(c).str.to_integer(base=16, strict=False).cast(pl.Float32)
                    / BYTE_MAX
                )
                .fill_null(PAD_SENTINEL)
                .alias(f"{c}_norm")
            )

        df = df.with_columns(payload_exprs)

        n = len(df)

        #
        # Numeric value matrix: (N, 10) -- [ID, DLC, Byte0..Byte7]
        #
        numeric_values = np.column_stack(
            [
                df["id_norm"].to_numpy(),
                df["dlc_norm"].to_numpy(),
                *[df[f"Data_{i}_norm"].to_numpy() for i in range(8)],
            ]
        ).astype(np.float32)

        #
        # Position
        #
        positions = np.broadcast_to(
            np.arange(
                10,
                dtype=np.int32,
            ),
            (n, 10),
        ).copy()

        #
        # Labels
        #
        labels = df["Class"].to_numpy().astype(np.int32)

        self.x = {
            "numeric_values": numeric_values,
            "positions": positions,
        }

        self.y = labels

    def to_tf_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(
            (
                self.x,
                self.y,
            )
        )

        if self.shuffle:
            ds = ds.shuffle(
                min(
                    len(self.y),
                    10000,
                ),
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(self.batch_size)

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
