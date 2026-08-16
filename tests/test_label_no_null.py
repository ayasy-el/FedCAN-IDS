"""
Pastikan tidak ada label yang null setelah tahap ingest -- terutama untuk
file raw yang tidak punya kolom SubClass sama sekali (pure normal
driving), yang seharusnya sudah di-fill jadi "Normal" sebelum encoding.

Jalankan setelah src/data/ingest.py dieksekusi:

    pytest tests/test_label_no_null.py
"""

from pathlib import Path

import polars as pl
import pytest

INTERIM_PATH = Path("data/interim/car_hacking_with_session.parquet")


def test_no_null_class_label():
    if not INTERIM_PATH.exists():
        pytest.skip(f"{INTERIM_PATH} belum ada, jalankan src/data/ingest.py dulu")

    df = pl.read_parquet(INTERIM_PATH)

    assert df["Class"].null_count() == 0, (
        "Ada baris dengan label Class null -- kemungkinan file raw tanpa "
        "kolom SubClass belum di-fill_null('Normal') sebelum encoding"
    )


def test_no_null_session_id():
    if not INTERIM_PATH.exists():
        pytest.skip(f"{INTERIM_PATH} belum ada, jalankan src/data/ingest.py dulu")

    df = pl.read_parquet(INTERIM_PATH)

    assert df["session_id"].null_count() == 0, "Ada baris tanpa session_id"
