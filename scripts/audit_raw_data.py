"""
Audit dataset raw sebelum masuk pipeline ingest.

Cek dua hal krusial per file capture:
1. Rentang Timestamp -- untuk mendeteksi apakah timestamp relatif
   (mulai dari ~0) atau epoch absolut, dan apakah antar file konsisten.
   Kalau tidak konsisten, WAJIB partisi per session_id di semua operasi
   urutan-sensitif (lihat src/data/feature.py, src/data/temporal_dataset.py).
2. Distribusi SubClass per file -- untuk mendeteksi file yang tidak
   punya kolom SubClass sama sekali (biasanya file pure normal driving).

Jalankan dari root project:

    python scripts/audit_raw_data.py
"""

from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw/Car_Hacking_Challenge_Dataset")


def audit():
    files = sorted(RAW_DIR.rglob("*.csv"))

    if not files:
        print(f"Tidak ada file CSV ditemukan di {RAW_DIR}")
        return

    for f in files:
        print(f"\n{f.name}")

        lazy = pl.scan_csv(f)
        columns = lazy.collect_schema().names()

        if "SubClass" not in columns:
            df = lazy.select("Timestamp", "Class").collect()
            print(
                f"Timestamp min: {df['Timestamp'].min()}  max: {df['Timestamp'].max()}"
            )
            print("SubClass: tidak ada")
            print(df["Class"].value_counts())
            continue

        df = lazy.select("Timestamp", "SubClass").collect()

        print(f"Timestamp min: {df['Timestamp'].min()}  max: {df['Timestamp'].max()}")
        print(df["SubClass"].value_counts())


if __name__ == "__main__":
    audit()
