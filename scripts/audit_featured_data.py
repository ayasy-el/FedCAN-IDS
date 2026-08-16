"""
Audit hasil feature engineering (output src/data/feature.py) sebelum
dipakai training.

Mengecek 4 hal utama, semuanya terkait langsung dengan bug yang pernah
ditemukan di pipeline ini:

1. Integritas session boundary -- delta_t/jitter TIDAK BOLEH dihitung
   lintas session_id. Deteksi tidak langsung: baris pertama tiap sesi
   (setelah diurutkan session_id + Timestamp) harus delta_t == 0 dan
   jitter == 0, karena diff()/shift() di baris pertama grup selalu null
   -> fill_null(0.0). Kalau ada yang bukan 0, berarti ada kebocoran
   lintas sesi.

2. Sanity value fitur temporal -- delta_t tidak boleh negatif (data
   sudah terurut ascending per sesi), dan tidak boleh ada outlier
   ekstrem yang mengindikasikan diff() salah partisi.

3. Kelengkapan data -- null count di semua kolom fitur, dan session_id/
   row_id tidak boleh null atau duplikat.

4. Distribusi kelas per split -- ringkasan cepat untuk cross-check
   dengan output src/data/split.py.

Jalankan setelah src/data/feature.py dieksekusi:

    python scripts/audit_featured_data.py
"""

from pathlib import Path

import polars as pl

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(500)

FEATURED_DIR = Path("data/processed/car_hacking/featured")

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


def audit_split(name: str, path: Path):
    print(f"\n{'=' * 70}")
    print(f"SPLIT: {name}  ({path})")
    print("=" * 70)

    if not path.exists():
        print(f"!! File tidak ditemukan: {path}")
        return

    df = pl.read_parquet(path)
    n = len(df)
    print(f"n_rows: {n:,}")

    # -------------------------------------------------------------
    # 1. Kelengkapan kolom & null check
    # -------------------------------------------------------------
    print("\n-- Null check --")
    missing_cols = [c for c in TEMPORAL_FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"!! Kolom fitur tidak ditemukan: {missing_cols}")

    check_cols = ["session_id", "row_id"] + [
        c for c in TEMPORAL_FEATURE_COLUMNS if c in df.columns
    ]
    null_counts = df.select(
        [pl.col(c).null_count().alias(c) for c in check_cols]
    ).to_dicts()[0]

    any_null = False
    for col, count in null_counts.items():
        if count > 0:
            any_null = True
            print(f"  !! {col}: {count:,} null")
    if not any_null:
        print("  OK: tidak ada null di session_id, row_id, dan kolom fitur temporal.")

    # -------------------------------------------------------------
    # 2. row_id unik (tidak ada duplikasi baris)
    # -------------------------------------------------------------
    print("\n-- Duplikasi row_id --")
    if "row_id" in df.columns:
        n_unique = df["row_id"].n_unique()
        if n_unique != n:
            print(f"  !! row_id tidak unik: {n:,} baris, {n_unique:,} row_id unik")
        else:
            print(f"  OK: {n_unique:,} row_id semuanya unik.")
    else:
        print("  !! kolom row_id tidak ditemukan.")

    # -------------------------------------------------------------
    # 3. Integritas session boundary: baris pertama tiap sesi
    # -------------------------------------------------------------
    print("\n-- Integritas session boundary (delta_t/jitter baris pertama sesi) --")
    if "session_id" in df.columns and "delta_t" in df.columns:
        first_rows = (
            df.sort(["session_id", "Timestamp"])
            .group_by("session_id", maintain_order=True)
            .first()
        )

        bad_delta = first_rows.filter(pl.col("delta_t") != 0.0)
        if len(bad_delta) > 0:
            print(
                f"  !! {len(bad_delta)} sesi dengan delta_t baris pertama != 0 "
                f"-- indikasi delta_t dihitung lintas session_id:"
            )
            print(bad_delta.select(["session_id", "delta_t"]))
        else:
            print("  OK: semua sesi punya delta_t == 0 di baris pertamanya.")

        if "jitter" in df.columns:
            bad_jitter = first_rows.filter(pl.col("jitter") != 0.0)
            if len(bad_jitter) > 0:
                print(
                    f"  !! {len(bad_jitter)} sesi dengan jitter baris pertama != 0 "
                    f"-- indikasi jitter dihitung lintas session_id:"
                )
                print(bad_jitter.select(["session_id", "jitter"]))
            else:
                print("  OK: semua sesi punya jitter == 0 di baris pertamanya.")
    else:
        print("  !! kolom session_id/delta_t tidak ditemukan, skip cek ini.")

    # -------------------------------------------------------------
    # 4. Sanity value: delta_t tidak boleh negatif
    # -------------------------------------------------------------
    print("\n-- Sanity delta_t negatif --")
    if "delta_t" in df.columns:
        n_negative = df.filter(pl.col("delta_t") < 0.0).height
        if n_negative > 0:
            print(
                f"  !! {n_negative:,} baris dengan delta_t negatif "
                f"-- data mungkin tidak terurut per sesi sebelum diff()."
            )
        else:
            print("  OK: tidak ada delta_t negatif.")

    # -------------------------------------------------------------
    # 5. Distribusi (describe) tiap fitur temporal
    # -------------------------------------------------------------
    print("\n-- Distribusi fitur temporal --")
    available = [c for c in TEMPORAL_FEATURE_COLUMNS if c in df.columns]
    if available:
        print(df.select(available).describe())

    # -------------------------------------------------------------
    # 6. Distribusi kelas
    # -------------------------------------------------------------
    print("\n-- Distribusi kelas --")
    if "Class" in df.columns:
        print(df.group_by("Class").len().rename({"len": "count"}).sort("Class"))


def main():
    for split_name in ["train", "val", "test"]:
        audit_split(split_name, FEATURED_DIR / f"{split_name}.parquet")

    print(f"\n{'=' * 70}")
    print("Audit selesai.")
    print("=" * 70)


if __name__ == "__main__":
    main()
