"""
Pastikan fitur temporal (Deltatime, jitter, dst) tidak pernah dihitung
lintas session_id. Deteksi tidak langsung: baris pertama tiap session_id
(setelah diurutkan per session_id + Timestamp) harus punya Deltatime == 0,
karena diff() di baris pertama grup selalu null -> fill_null(0.0).

Jalankan setelah src/data/feature.py dieksekusi:

    pytest tests/test_feature_no_cross_session.py
"""

from pathlib import Path

import polars as pl
import pytest

FEATURED_DIR = Path("data/processed/car_hacking/featured")


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_first_row_per_session_has_zero_deltatime(split_name):
    path = FEATURED_DIR / f"{split_name}.parquet"

    if not path.exists():
        pytest.skip(f"{path} belum ada, jalankan src/data/feature.py dulu")

    df = pl.read_parquet(path).sort(["session_id", "Timestamp"])

    first_rows = df.group_by("session_id", maintain_order=True).first()

    non_zero = first_rows.filter(pl.col("Deltatime") != 0.0)

    assert len(non_zero) == 0, (
        "Ada baris pertama sesi dengan Deltatime != 0 -- indikasi Deltatime "
        "dihitung lintas session_id (bug #2 belum sepenuhnya diperbaiki)"
    )


@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_no_null_session_id(split_name):
    path = FEATURED_DIR / f"{split_name}.parquet"

    if not path.exists():
        pytest.skip(f"{path} belum ada, jalankan src/data/feature.py dulu")

    df = pl.read_parquet(path)

    assert df["session_id"].null_count() == 0, "Ada baris tanpa session_id"
