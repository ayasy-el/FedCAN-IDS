"""Task labels and model/data compatibility contracts."""

LABEL_SCHEMAS = {
    "binary": {
        "num_classes": 2,
        "class_names": ["Normal", "Attack"],
        "benign_labels": [0],
    },
    "five_class": {
        "num_classes": 5,
        "class_names": ["Normal", "Flooding", "Fuzzing", "Spoofing", "Replay"],
        "benign_labels": [0],
    },
    "ten_state_attack": {
        "num_classes": 10,
        "class_names": [
            "benign-driving", "DoS-driving", "Fuzzing-driving", "Spoofing-driving", "Replay-driving",
            "benign-stationary", "DoS-stationary", "Fuzzing-stationary", "Spoofing-stationary", "Replay-stationary",
        ],
        "benign_labels": [0, 5],
    },
    "CICIoV2024": {
        "num_classes": 6,
        "class_names": ["BENIGN", "DoS", "GAS", "RPM", "SPEED", "STEERING_WHEEL"],
        "benign_labels": [0],
    },
    "SurvivalAnalysis": {
        "num_classes": 4,
        "class_names": ["Normal", "Flooding", "Fuzzing", "Malfunction"],
        "benign_labels": [0],
    },
    "HCRLCarHacking": {
        "num_classes": 5,
        "class_names": ["Normal", "DoS", "Fuzzy", "Gear", "RPM"],
        "benign_labels": [0],
    },
}

CAN_MIRGU_SCHEMAS = {
    "real": {
        "num_classes": 5,
        "class_names": ["Normal", "DoS", "Fuzzing", "Spoofing", "Replay"],
        "benign_labels": [0],
    },
    "extended": {
        "num_classes": 7,
        "class_names": [
            "Normal", "DoS", "Fuzzing", "Spoofing", "Replay",
            "Masquerade", "Suspension",
        ],
        "benign_labels": [0],
    },
}

ROAD_SCHEMAS = {
    "fabrication": {
        "num_classes": 7,
        "class_names": ["Normal", "MEC", "Fuzzing", "MS", "RLOn", "RLOff", "CS"],
        "benign_labels": [0],
    },
    "masquerade": {
        "num_classes": 6,
        "class_names": ["Normal", "MEC", "MS", "RLOn", "RLOff", "CS"],
        "benign_labels": [0],
    },
}

MODEL_CONTRACTS = {
    "mlp": {"unit": "frame"},
    "mlp_window": {"unit": "window"},
    "can_bigrubert": {"unit": "window"},
}


def label_schema(name, variant=None):
    if name == "CANMIRGU":
        try:
            return CAN_MIRGU_SCHEMAS[variant]
        except KeyError as exc:
            raise ValueError(
                "label_schema='CANMIRGU' requires dataset.variant='real' "
                "or 'extended'"
            ) from exc
    if name == "road":
        try:
            return ROAD_SCHEMAS[variant]
        except KeyError as exc:
            raise ValueError(
                "label_schema='road' requires dataset.variant='fabrication' "
                "or 'masquerade'"
            ) from exc
    try:
        return LABEL_SCHEMAS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown label schema {name!r}; choose one of "
            f"{sorted((*LABEL_SCHEMAS, 'road', 'CANMIRGU'))}"
        ) from exc


def validate_experiment(params):
    experiment = params.get("experiment", {})
    model_name = experiment.get("model")
    split = params.get("split", {})
    schema = label_schema(
        params.get("prepare", {}).get("label_schema", "five_class"),
        params.get("dataset", {}).get("variant"),
    )
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
