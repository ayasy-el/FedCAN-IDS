"""Create shared frame features and the configured label representation."""

import os
import tempfile
from pathlib import Path

import polars as pl

from data.task import label_schema


ATTACK_IDS = {"Normal": 0, "Flooding": 1, "Fuzzing": 2, "Spoofing": 3, "Replay": 4}
CICIOV2024_IDS = {
    "BENIGN": 0,
    "DoS": 1,
    "GAS": 2,
    "RPM": 3,
    "SPEED": 4,
    "STEERING_WHEEL": 5,
}
SURVIVAL_ANALYSIS_IDS = {
    "Normal": 0,
    "Flooding": 1,
    "Fuzzing": 2,
    "Malfunction": 3,
}
CAR_HACKING_SCENARIOS_IDS = {
    "Normal": 0,
    "DoS": 1,
    "Fuzzy": 2,
    "Gear": 3,
    "RPM": 4,
}
ROAD_IDS = {
    "Normal": 0,
    "MEC": 1,
    "Fuzzing": 2,
    "MS": 3,
    "RLOn": 4,
    "RLOff": 5,
    "CS": 6,
}
ROAD_MASQUERADE_IDS = {
    "Normal": 0,
    "MEC": 1,
    "MS": 2,
    "RLOn": 3,
    "RLOff": 4,
    "CS": 5,
}
CAN_MIRGU_IDS = {
    "Normal": 0,
    "DoS": 1,
    "Fuzzing": 2,
    "Spoofing": 3,
    "Replay": 4,
    "Masquerade": 5,
    "Suspension": 6,
}


def _state_from_session(session):
    if "_D_" in session or session.endswith("_D"):
        return "D"
    if "_S_" in session or session.endswith("_S"):
        return "S"
    raise ValueError(f"Cannot infer vehicle state from session name: {session}")


def _label_expr(schema_name, variant=None):
    attack = pl.col("attack_type")
    attack_id = attack.replace_strict(ATTACK_IDS).cast(pl.UInt8)
    if schema_name == "binary":
        return (attack != "Normal").cast(pl.UInt8)
    if schema_name == "five_class":
        return attack_id
    if schema_name == "ten_state_attack":
        stationary = pl.col("session_id").str.contains(r"_S(?:_|$)")
        return (attack_id + stationary.cast(pl.UInt8) * 5).cast(pl.UInt8)
    if schema_name == "CICIoV2024":
        return pl.col("attack_type").replace_strict(CICIOV2024_IDS).cast(pl.UInt8)
    if schema_name == "SurvivalAnalysis":
        return pl.col("attack_type").replace_strict(SURVIVAL_ANALYSIS_IDS).cast(pl.UInt8)
    if schema_name == "HCRLCarHacking":
        return pl.col("attack_type").replace_strict(CAR_HACKING_SCENARIOS_IDS).cast(pl.UInt8)
    if schema_name == "road":
        if variant == "fabrication":
            return pl.col("attack_type").replace_strict(ROAD_IDS).cast(pl.UInt8)
        if variant == "masquerade":
            return pl.col("attack_type").replace_strict(ROAD_MASQUERADE_IDS).cast(pl.UInt8)
        raise ValueError("ROAD requires variant='fabrication' or 'masquerade'")
    if schema_name == "CANMIRGU":
        schema = label_schema(schema_name, variant)
        allowed = {name: index for index, name in enumerate(schema["class_names"])}
        return pl.col("attack_type").replace_strict(allowed).cast(pl.UInt8)
    label_schema(schema_name, variant)
    raise AssertionError("unreachable")


def run(params):
    dataset = params["dataset"]
    source = Path(dataset["interim_path"])
    output = Path(dataset["prepared_path"])
    schema_name = params.get("prepare", {}).get("label_schema", "five_class")
    variant = dataset.get("variant")
    requested_features = params.get("prepare", {}).get(
        "features", ["deltatime", "delta_id"]
    )
    feature_names = {
        "deltatime": "Deltatime",
        "delta_id": "Delta_Id",
    }
    unknown_features = set(requested_features) - set(feature_names)
    if unknown_features:
        raise ValueError(
            f"Unknown prepare.features: {sorted(unknown_features)}; "
            f"choose from {sorted(feature_names)}"
        )
    label_schema(schema_name, variant)

    streaming_adapter = dataset.get("ingest_adapter") in {
        "road_raw",
        "can_mirgu",
        "hcrl_carhacking",
        "ciciov2024",
        "car_hacking_attack_defense",
        "survival_analysis",
    }
    if streaming_adapter:
        # Streaming adapters write one session per file, in timestamp/file order.
        # A global sort here would materialize the whole dataset and cause OOM.
        frame_data = pl.scan_parquet(source)
        source_columns = frame_data.collect_schema().names()
    else:
        frame_data = pl.read_parquet(source)
        sort_columns = ["session_id", "Timestamp"] if "Timestamp" in frame_data.columns else ["session_id", "row_id"]
        frame_data = frame_data.sort(sort_columns)
        source_columns = frame_data.columns

    if requested_features and "Timestamp" not in source_columns:
        raise ValueError(
            "Requested timing features require Timestamp, but the ingested dataset "
            "does not provide one. Set prepare.features: [] or use a dataset with timestamps."
        )
    expressions = [_label_expr(schema_name, variant).alias("Class")]
    if "deltatime" in requested_features:
        expressions.append(
            pl.col("Timestamp")
            .diff()
            .over("session_id")
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("Deltatime")
        )
    if "delta_id" in requested_features:
        expressions.append(
            pl.col("Timestamp")
            .diff()
            .over(["session_id", "Arbitration_ID"])
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias("Delta_Id")
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if streaming_adapter:
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        temporary.close()
        try:
            (
                frame_data.with_columns(expressions)
                .drop("attack_type")
                .sink_parquet(
                    temporary_path,
                    compression="zstd",
                    row_group_size=100_000,
                    maintain_order=True,
                    engine="streaming",
                )
            )
            os.replace(temporary_path, output)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        count = (
            pl.scan_parquet(output)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
    else:
        result = frame_data.with_columns(expressions).drop("attack_type")
        result.write_parquet(output, compression="zstd")
        count = len(result)
    print(f"Prepared {count:,} frames with label schema {schema_name!r}")


if __name__ == "__main__":
    from utils.params import load_params

    run(load_params())
