from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf


PAD_VALUE = 256


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

        self.df = pl.read_parquet(
            self.parquet_path
        )

        self.prepare()

    def prepare(self):
        df = self.df

        #
        # Arbitration_ID
        # "4F1" -> 1265
        #
        df = df.with_columns(
            pl.col("Arbitration_ID")
            .str.to_integer(
                base=16,
                strict=False,
            )
            .fill_null(0)
            .cast(pl.UInt32)
            .alias("Arbitration_ID")
        )

        #
        # Payload bytes
        # "FF" -> 255
        # "PAD" -> 256
        #
        payload_exprs = []

        for i in range(8):
            c = f"Data_{i}"

            payload_exprs.append(
                pl.col(c)
                .str.to_integer(
                    base=16,
                    strict=False,
                )
                .fill_null(PAD_VALUE)
                .cast(pl.UInt16)
                .alias(c)
            )

        df = df.with_columns(
            payload_exprs
        )

        #
        # Token matrix
        #
        tokens = np.column_stack(
            [
                df["Arbitration_ID"]
                .to_numpy()
                .astype(np.int32),

                df["DLC"]
                .to_numpy()
                .astype(np.int32),

                *[
                    df[f"Data_{i}"]
                    .to_numpy()
                    .astype(np.int32)
                    for i in range(8)
                ],
            ]
        )

        n = len(df)

        #
        # Token type
        #
        # 0 = ID
        # 1 = DLC
        # 2 = BYTE
        # 3 = PAD
        #
        token_types = np.full(
            (n, 10),
            2,
            dtype=np.int32,
        )

        token_types[:, 0] = 0
        token_types[:, 1] = 1

        payload = tokens[:, 2:]

        token_types[:, 2:] = np.where(
            payload == PAD_VALUE,
            3,
            2,
        )

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
        labels = (
            df["Class"]
            .to_numpy()
            .astype(np.int32)
        )

        self.x = {
            "tokens": tokens,
            "token_types": token_types,
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

        ds = ds.batch(
            self.batch_size
        )

        ds = ds.prefetch(
            tf.data.AUTOTUNE
        )

        return ds
