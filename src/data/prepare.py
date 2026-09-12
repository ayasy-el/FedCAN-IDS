"""Create shared frame features and the configured label representation."""

from pathlib import Path

import polars as pl

from data.task import label_schema


ATTACK_IDS = {"Normal": 0, "Flooding": 1, "Fuzzing": 2, "Spoofing": 3, "Replay": 4}


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
    label_schema(schema_name)
    raise AssertionError("unreachable")


def run(params):
    dataset = params["dataset"]
    source = Path(dataset["interim_path"])
    output = Path(dataset["prepared_path"])
    schema_name = params.get("prepare", {}).get("label_schema", "five_class")
    label_schema(schema_name)
    df = pl.read_parquet(source).sort(["session_id", "Timestamp"])
    result = df.with_columns(
        pl.col("Timestamp").diff().over("session_id").fill_null(0.0).cast(pl.Float32).alias("Deltatime"),
        pl.col("Timestamp").diff().over(["session_id", "Arbitration_ID"]).fill_null(0.0).cast(pl.Float32).alias("Delta_Id"),
        _label_expr(schema_name).alias("Class"),
    ).drop("attack_type")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd")
    print(f"Prepared {len(result):,} frames with label schema {schema_name!r}")


if __name__ == "__main__":
    from utils.params import load_params

    run(load_params())
