"""Unit tests for dataset ingestion adapters."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from data.adapters import (
    can_mirgu,
    car_hacking_attack_defense,
    ciciov2024,
    hcrl_carhacking,
    survival_analysis,
)


EXPECTED_COLUMNS = {
    "row_id",
    "session_id",
    "Arbitration_ID",
    "DLC",
    *[f"Data_{i}" for i in range(8)],
    "attack_type",
}


def test_hcrl_carhacking_streaming():
    csv_file = Path("data/raw/9) Car-Hacking Dataset/DoS_dataset.csv")
    txt_file = Path("data/raw/9) Car-Hacking Dataset/normal_run_data/normal_run_data.txt")
    if not csv_file.exists() or not txt_file.exists():
        pytest.skip("HCRL Car-Hacking raw files not found")

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        out_path = Path(tmp.name)
        # Test write on attack CSV and normal TXT
        total_rows = hcrl_carhacking.write([csv_file, txt_file], out_path)
        assert total_rows > 0
        assert out_path.exists()

        df = pl.scan_parquet(out_path)
        schema = df.collect_schema()
        assert EXPECTED_COLUMNS.issubset(set(schema.names()))
        assert "Timestamp" in schema.names()

        # Check sample
        head_sample = df.head(10).collect()
        assert head_sample["row_id"].to_list() == list(range(10))
        assert head_sample["Arbitration_ID"].str.len_chars().max() == 3
        assert head_sample["DLC"].dtype == pl.UInt8


def test_ciciov2024_streaming():
    raw_dir = Path("data/raw/CICIoV2024/hexadecimal")
    if not raw_dir.exists():
        pytest.skip("CICIoV2024 raw dir not found")
    files = sorted(raw_dir.glob("*.csv"))[:2]
    if not files:
        pytest.skip("CICIoV2024 CSV files not found")

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        out_path = Path(tmp.name)
        total_rows = ciciov2024.write(files, out_path)
        assert total_rows > 0
        assert out_path.exists()

        df = pl.scan_parquet(out_path)
        schema = df.collect_schema()
        assert EXPECTED_COLUMNS.issubset(set(schema.names()))

        head_sample = df.head(10).collect()
        assert head_sample["row_id"].to_list() == list(range(10))
        assert head_sample["Arbitration_ID"].str.len_chars().max() == 3


def test_car_hacking_attack_defense_streaming():
    raw_dir = Path("data/raw/Car_Hacking_Challenge_Dataset/0_Preliminary/0_Training")
    if not raw_dir.exists():
        pytest.skip("Car Hacking Challenge dataset not found")
    files = sorted(raw_dir.glob("*.csv"))[:2]
    if not files:
        pytest.skip("Car Hacking Challenge CSV files not found")

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        out_path = Path(tmp.name)
        total_rows = car_hacking_attack_defense.write(files, out_path)
        assert total_rows > 0
        assert out_path.exists()

        df = pl.scan_parquet(out_path)
        schema = df.collect_schema()
        assert EXPECTED_COLUMNS.issubset(set(schema.names()))
        assert "Timestamp" in schema.names()

        head_sample = df.head(10).collect()
        assert head_sample["row_id"].to_list() == list(range(10))
        assert head_sample["Arbitration_ID"].str.len_chars().max() == 3


def test_survival_analysis_streaming():
    raw_dir = Path("data/raw/Survival Analysis Dataset/dataset")
    if not raw_dir.exists():
        pytest.skip("Survival Analysis raw dir not found")
    files = sorted(raw_dir.rglob("*.txt"))[:2]
    if not files:
        pytest.skip("Survival Analysis TXT files not found")

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        out_path = Path(tmp.name)
        total_rows = survival_analysis.write(files, out_path)
        assert total_rows > 0
        assert out_path.exists()

        df = pl.scan_parquet(out_path)
        schema = df.collect_schema()
        assert EXPECTED_COLUMNS.issubset(set(schema.names()))
        assert "Timestamp" in schema.names()

        head_sample = df.head(10).collect()
        assert head_sample["row_id"].to_list() == list(range(10))
        assert head_sample["Arbitration_ID"].str.len_chars().max() == 3
