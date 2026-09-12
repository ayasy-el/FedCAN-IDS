"""Task labels and model/data compatibility contracts."""

LABEL_SCHEMAS = {
    "binary": {
        "num_classes": 2,
        "class_names": ["Normal", "Attack"],
    },
    "five_class": {
        "num_classes": 5,
        "class_names": ["Normal", "Flooding", "Fuzzing", "Spoofing", "Replay"],
    },
    "ten_state_attack": {
        "num_classes": 10,
        "class_names": [
            "benign-driving", "DoS-driving", "Fuzzing-driving", "Spoofing-driving", "Replay-driving",
            "benign-stationary", "DoS-stationary", "Fuzzing-stationary", "Spoofing-stationary", "Replay-stationary",
        ],
    },
}

MODEL_CONTRACTS = {
    "mlp": {"unit": "frame"},
    "mlp_window": {"unit": "window"},
    "can_bigrubert": {"unit": "window"},
}


def label_schema(name):
    try:
        return LABEL_SCHEMAS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown label schema {name!r}; choose one of {sorted(LABEL_SCHEMAS)}"
        ) from exc


def validate_experiment(params):
    experiment = params.get("experiment", {})
    model_name = experiment.get("model")
    split = params.get("split", {})
    schema = label_schema(params.get("prepare", {}).get("label_schema", "five_class"))
    if model_name not in MODEL_CONTRACTS:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {sorted(MODEL_CONTRACTS)}")
    unit = split.get("unit", "frame")
    required_unit = MODEL_CONTRACTS[model_name]["unit"]
    if unit != required_unit:
        raise ValueError(
            f"Model {model_name!r} requires unit={required_unit!r}, but split.unit={unit!r}"
        )
    configured = params.get("model", {}).get(model_name, {}).get("num_classes")
    if configured is not None and int(configured) != schema["num_classes"]:
        raise ValueError(
            f"model.{model_name}.num_classes must match prepare.label_schema "
            f"({schema['num_classes']}), got {configured}"
        )
    return model_name, schema
