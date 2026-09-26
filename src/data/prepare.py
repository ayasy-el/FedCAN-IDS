"""Create shared frame features and the configured label representation."""

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


def _state_from_session(session):
    if "_D_" in session or session.endswith("_D"):
        return "D"
    if "_S_" in session or session.endswith("_S"):
        return "S"
    raise ValueError(f"Cannot infer vehicle state from session name: {session}")


def _label_expr(schema_name):
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
    label_schema(schema_name)
    raise AssertionError("unreachable")


def run(params):
    dataset = params["dataset"]
    source = Path(dataset["interim_path"])
    output = Path(dataset["prepared_path"])
    schema_name = params.get("prepare", {}).get("label_schema", "five_class")
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
    label_schema(schema_name)
    df = pl.read_parquet(source)
    sort_columns = ["session_id", "Timestamp"] if "Timestamp" in df.columns else ["session_id", "row_id"]
    df = df.sort(sort_columns)
    if requested_features and "Timestamp" not in df.columns:
        raise ValueError(
            "Requested timing features require Timestamp, but the ingested dataset "
            "does not provide one. Set prepare.features: [] or use a dataset with timestamps."
        )
    expressions = [_label_expr(schema_name).alias("Class")]
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
    result = df.with_columns(expressions).drop("attack_type")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd")
    print(f"Prepared {len(result):,} frames with label schema {schema_name!r}")


if __name__ == "__main__":
    from utils.params import load_params

    run(load_params())
