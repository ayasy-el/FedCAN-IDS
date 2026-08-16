"""
Pastikan split.py tidak membocorkan baris yang sama ke lebih dari satu
split. Setiap session_id BOLEH muncul di lebih dari satu split (karena
strategi split memotong tiap sesi jadi 3 segmen kronologis: train/val/
test) -- yang tidak boleh adalah baris (row_id) yang sama muncul dobel.

Jalankan setelah src/data/split.py dieksekusi:

    pytest tests/test_split_no_session_leak.py
"""

from pathlib import Path

import polars as pl
import pytest

PROCESSED_DIR = Path("data/processed/car_hacking")


@pytest.fixture(scope="module")
def splits():
    train = pl.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pl.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pl.read_parquet(PROCESSED_DIR / "test.parquet")
    return train, val, test


def test_no_duplicate_row_id_across_splits(splits):
    train, val, test = splits

    train_ids = set(train["row_id"].to_list())
    val_ids = set(val["row_id"].to_list())
    test_ids = set(test["row_id"].to_list())

    assert len(train_ids & val_ids) == 0, "Ada row_id yang bocor antara train & val"
    assert len(train_ids & test_ids) == 0, "Ada row_id yang bocor antara train & test"
    assert len(val_ids & test_ids) == 0, "Ada row_id yang bocor antara val & test"


def test_every_class_present_in_val_and_test(splits):
    train, val, test = splits

    train_classes = set(train["Class"].unique().to_list())
    val_classes = set(val["Class"].unique().to_list())
    test_classes = set(test["Class"].unique().to_list())

    missing_in_val = train_classes - val_classes
    missing_in_test = train_classes - test_classes

    assert not missing_in_val, f"Kelas hilang total di val: {missing_in_val}"
    assert not missing_in_test, f"Kelas hilang total di test: {missing_in_test}"
