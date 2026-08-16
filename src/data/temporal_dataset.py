import json
from pathlib import Path

import numpy as np
import polars as pl
import tensorflow as tf


# Urutan kolom fitur temporal -- HARUS konsisten dengan urutan yang
# dipakai saat menghitung/menyimpan statistik normalisasi (mean/std),
# supaya index kolom di file stats JSON tidak pernah tertukar dengan
# index kolom saat dipakai kembali.
TEMPORAL_FEATURE_COLUMNS = [
    "delta_t",
    "delta_t_z",
    "jitter",
    "hamming_distance",
    "frame_rate_20ms",
    "id_rate_20ms",
    "id_entropy_20ms",
    "dominant_id_ratio_20ms",
    "bus_load_proxy_20ms",
]


class TemporalCANDataset:
    """
    Output:

    (
        {
            "tokens": (T,10),
            "token_types": (T,10),
            "positions": (T,10),
            "temporal_features": (T,9),
        },
        label
    )
    """

    def __init__(
        self,
        parquet_path,
        seq_len=32,
        batch_size=256,
        shuffle=True,
        shuffle_buffer=10000,
        normalize_stats_path=None,
        fit_normalize_stats=False,
    ):
        self.parquet_path = parquet_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.normalize_stats_path = normalize_stats_path
        self.fit_normalize_stats = fit_normalize_stats

        self.prepare()

    # =====================================================
    # Prepare dataframe
    # =====================================================

    def prepare(self):

        df = (
            pl.read_parquet(self.parquet_path)
            # PENTING: sort per session_id dulu, baru Timestamp di dalamnya.
            # Timestamp antar sesi capture TIDAK sinkron (sebagian relatif
            # ke awal capture, sebagian epoch absolut) -- sort("Timestamp")
            # global akan mengurutkan seluruh dataset seolah-olah satu
            # timeline tunggal, padahal itu bisa mencampur baris dari sesi
            # yang berbeda-beda secara acak.
            .sort(["session_id", "Timestamp"])
        )

        #
        # Arbitration ID
        # "4F1" -> 1265
        #
        df = df.with_columns(
            pl.col("Arbitration_ID").map_elements(
                lambda x: int(x, 16),
                return_dtype=pl.UInt16,
            )
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
                .map_elements(
                    lambda x: 256 if x == "PAD" else int(x, 16),
                    return_dtype=pl.UInt16,
                )
                .alias(c)
            )

        df = df.with_columns(payload_exprs)

        #
        # Token matrix
        #
        self.tokens = np.column_stack(
            [
                df["Arbitration_ID"].to_numpy(),
                df["DLC"].to_numpy(),
                *[df[f"Data_{i}"].to_numpy() for i in range(8)],
            ]
        ).astype(np.int32)

        #
        # Temporal features (mentah, sebelum normalisasi)
        #
        self.temporal = (
            df.select(TEMPORAL_FEATURE_COLUMNS)
            .fill_null(0)
            .to_numpy()
            .astype(np.float32)
        )

        #
        # Normalisasi z-score (fix baru -- lihat docstring kelas di atas)
        #
        self._apply_temporal_normalization()

        #
        # Labels
        #
        self.labels = df["Class"].to_numpy().astype(np.int32)

        #
        # session_id per baris (dipakai generator() untuk memastikan satu
        # window seq_len tidak menyeberang batas sesi capture -- window
        # yang berisi baris dari >1 sesi tidak merepresentasikan traffic
        # CAN yang benar-benar berurutan, jadi harus di-skip)
        #
        self.session_ids = df["session_id"].to_numpy()

        #
        # Constant matrices
        #

        self.token_types = np.array(
            [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            dtype=np.int32,
        )

        self.positions = np.arange(
            10,
            dtype=np.int32,
        )

        # num_samples dihitung dari window yang benar-benar VALID (tidak
        # menyeberang batas session_id), bukan sekadar len(labels)-seq_len,
        # supaya shuffle buffer size representatif dan estimasi jumlah
        # sequence akurat.
        T = self.seq_len
        n = len(self.labels)

        if n > T:
            starts = np.arange(T, n)
            valid = self.session_ids[starts - T] == self.session_ids[starts - 1]
            self.num_samples = int(valid.sum())

            # Label per window yang VALID -- dipakai untuk menghitung
            # class_weight yang merepresentasikan distribusi label
            # SEBENARNYA yang dilihat model saat training (bukan
            # distribusi label per-frame mentah, yang sedikit berbeda
            # karena window di batas sesi di-skip).
            self.window_labels = self.labels[starts][valid]
        else:
            self.num_samples = 0
            self.window_labels = np.array([], dtype=np.int32)

        print(f"Frames   : {len(self.labels):,}")
        print(f"Sequences: {self.num_samples:,}")

    # =====================================================
    # Normalisasi fitur temporal
    # =====================================================

    def _apply_temporal_normalization(self):

        if self.normalize_stats_path is None:
            print(
                "PERINGATAN: normalize_stats_path tidak diberikan -- "
                "fitur temporal TIDAK dinormalisasi. Ini kemungkinan "
                "menyebabkan training tidak stabil (lihat docstring "
                "TemporalCANDataset)."
            )
            return

        stats_path = Path(self.normalize_stats_path)

        if self.fit_normalize_stats:
            mean = self.temporal.mean(axis=0)
            std = self.temporal.std(axis=0)

            # Hindari pembagian dengan nol untuk kolom yang kebetulan
            # konstan (std=0) di split train.
            std_safe = np.where(std < 1e-8, 1.0, std)

            stats_path.parent.mkdir(parents=True, exist_ok=True)

            with open(stats_path, "w") as f:
                json.dump(
                    {
                        "columns": TEMPORAL_FEATURE_COLUMNS,
                        "mean": mean.tolist(),
                        "std": std_safe.tolist(),
                        "computed_from": str(self.parquet_path),
                    },
                    f,
                    indent=2,
                )

            print(
                f"Statistik normalisasi (mean/std) dihitung dari "
                f"{self.parquet_path} dan disimpan ke {stats_path}"
            )

        else:
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"{stats_path} tidak ditemukan. Jalankan dataset TRAIN "
                    f"dengan fit_normalize_stats=True terlebih dahulu "
                    f"supaya statistik normalisasi tersimpan, baru pakai "
                    f"file itu untuk val/test. Val/test TIDAK BOLEH "
                    f"menghitung statistik normalisasi sendiri -- itu "
                    f"data leakage."
                )

            with open(stats_path) as f:
                stats = json.load(f)

            if stats["columns"] != TEMPORAL_FEATURE_COLUMNS:
                raise ValueError(
                    f"Urutan kolom fitur temporal di {stats_path} "
                    f"({stats['columns']}) tidak cocok dengan urutan yang "
                    f"dipakai saat ini ({TEMPORAL_FEATURE_COLUMNS}). "
                    f"Kemungkinan feature.py berubah urutan kolom sejak "
                    f"stats ini disimpan -- hitung ulang stats dari train."
                )

            mean = np.array(stats["mean"], dtype=np.float32)
            std_safe = np.array(stats["std"], dtype=np.float32)

            print(
                f"Statistik normalisasi dimuat dari {stats_path} (BUKAN "
                f"dihitung ulang dari {self.parquet_path}) -- mencegah "
                f"data leakage dari val/test."
            )

        self.temporal = (self.temporal - mean) / std_safe

    # =====================================================
    # Generator
    # =====================================================

    def generator(self):

        T = self.seq_len

        for i in range(
            T,
            len(self.labels),
        ):
            # Skip window yang mengandung baris dari lebih dari satu
            # session_id -- window seperti ini bukan traffic CAN yang
            # benar-benar berurutan (menyambung dua sesi capture berbeda),
            # jadi delta_t/jitter/window-features di dalamnya tidak valid.
            # Karena data sudah di-sort per session_id (blok kontigu),
            # cukup bandingkan ujung awal & akhir window.
            if self.session_ids[i - T] != self.session_ids[i - 1]:
                continue

            yield (
                {
                    "tokens": self.tokens[i - T : i],
                    "token_types": np.broadcast_to(self.token_types, (T, 10)),
                    "positions": np.broadcast_to(self.positions, (T, 10)),
                    "temporal_features": self.temporal[i - T : i],
                },
                self.labels[i],
            )

    # =====================================================
    # Tensorflow Dataset
    # =====================================================

    def to_tf_dataset(self):

        output_signature = (
            {
                "tokens": tf.TensorSpec(
                    shape=(
                        self.seq_len,
                        10,
                    ),
                    dtype=tf.int32,
                ),
                "token_types": tf.TensorSpec(
                    shape=(
                        self.seq_len,
                        10,
                    ),
                    dtype=tf.int32,
                ),
                "positions": tf.TensorSpec(
                    shape=(
                        self.seq_len,
                        10,
                    ),
                    dtype=tf.int32,
                ),
                "temporal_features": tf.TensorSpec(
                    shape=(
                        self.seq_len,
                        9,
                    ),
                    dtype=tf.float32,
                ),
            },
            tf.TensorSpec(
                shape=(),
                dtype=tf.int32,
            ),
        )

        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature,
        )

        if self.shuffle:
            ds = ds.shuffle(
                min(
                    self.shuffle_buffer,
                    self.num_samples,
                )
            )

        ds = ds.batch(
            self.batch_size,
            drop_remainder=False,
        )

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds
