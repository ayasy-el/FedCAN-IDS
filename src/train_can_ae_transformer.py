"""Train the CAN-AE-Transformer model."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import dagshub
import mlflow
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import CosineDecay

from data.can_ae_transformer_dataset import CANAeTransformerDataset
from model.can_ae_transformer import (
    build_can_ae_transformer,
    parameter_counts,
)
from utils.metrics import MacroF1Score, MacroPrecision, MacroRecall
from utils.mlflow_utils import (
    BestEpochMetrics,
    MlflowEpochLogger,
    log_artifact_organized,
    save_run_id,
    write_model_report,
)
from utils.params import load_params
from data.task import validate_experiment

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)


# ==========================================================
# Load params.yaml
# ==========================================================

params = load_params()
dataset = params["dataset"]
_, task = validate_experiment(params)
num_classes = task["num_classes"]
split = params["split"]
model_params = load_params("model.can_ae_transformer")
training = load_params("training.can_ae_transformer")
mlflow_params = load_params("mlflow")
window_size = int(split["window_size"])

best_path = "checkpoints/can_ae_transformer_best.keras"
final_path = "checkpoints/can_ae_transformer_final.keras"
run_id_path = "checkpoints/can_ae_transformer_mlflow_run_id.txt"
best_metrics_path = "reports/metrics/can_ae_transformer_best_training_metrics.json"
report_path = "reports/model_summaries/can_ae_transformer_model_report.txt"


# ==========================================================
# Dataset
# ==========================================================

train = CANAeTransformerDataset(
    f"{dataset['processed_dir']}/train.parquet",
    window_size=window_size,
    d_model=model_params["d_model"],
    granularity=float(model_params.get("granularity", 1e-8)),
    max_time_position=int(model_params.get("max_time_position", 10000)),
    batch_size=training["batch_size"],
    shuffle=True,
    random_seed=split["random_seed"],
    source_path=dataset["prepared_path"],
)
val = CANAeTransformerDataset(
    f"{dataset['processed_dir']}/val.parquet",
    window_size=window_size,
    d_model=model_params["d_model"],
    granularity=float(model_params.get("granularity", 1e-8)),
    max_time_position=int(model_params.get("max_time_position", 10000)),
    batch_size=training["batch_size"],
    shuffle=False,
    random_seed=split["random_seed"],
    source_path=dataset["prepared_path"],
)
has_validation = len(val.y) > 0
monitor = "val_f1_macro" if has_validation else "f1_macro"


# ==========================================================
# Model and build
# ==========================================================

model = build_can_ae_transformer(
    window_size=window_size,
    d_model=model_params["d_model"],
    num_heads=model_params["num_heads"],
    num_layers=model_params["num_layers"],
    dim_feedforward=model_params["dim_feedforward"],
    dropout=model_params["dropout"],
    num_classes=num_classes,
)

# Cosine decay schedule with linear warmup
steps_per_epoch = max(1, len(train.y) // training["batch_size"])
total_steps = steps_per_epoch * training["epochs"]
warmup_steps = int(training.get("warmup_steps", 100))
lr_schedule = CosineDecay(
    initial_learning_rate=float(training["learning_rate"]),
    decay_steps=total_steps,
    warmup_steps=min(warmup_steps, max(1, total_steps // 10)),
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        MacroPrecision(num_classes),
        MacroRecall(num_classes),
        MacroF1Score(num_classes),
    ],
)
model.summary()

write_model_report(
    report_path,
    "can_ae_transformer",
    model,
    {
        "dataset": dataset,
        "task": task,
        "split": split,
        "model": model_params,
        "training": training,
        "parameter_counts": parameter_counts(model),
        "num_classes": num_classes,
    },
)


# ==========================================================
# MLflow tracking
# ==========================================================

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_can_ae_transformer"])
with mlflow.start_run() as run:
    save_run_id(run.info.run_id, run_id_path)
    mlflow.log_params({f"model.{key}": value for key, value in model_params.items()})
    mlflow.log_params({f"training.{key}": value for key, value in training.items()})
    mlflow.log_params({f"split.{key}": value for key, value in split.items()})
    mlflow.log_params(parameter_counts(model))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_path,
            monitor=monitor,
            mode="max",
            save_best_only=True,
        ),
        BestEpochMetrics(best_metrics_path, monitor=monitor, mode="max"),
        MlflowEpochLogger(),
    ]

    fit_kwargs = {"epochs": training["epochs"], "callbacks": callbacks}
    if has_validation:
        fit_kwargs["validation_data"] = val.to_tf_dataset()

    model.fit(train.to_tf_dataset(), **fit_kwargs)
    model.save(final_path)

    log_artifact_organized(best_path)
    log_artifact_organized(final_path)
    log_artifact_organized(best_metrics_path)
    log_artifact_organized(report_path, "reports/model_summaries")

print("CAN-AE-Transformer training finished successfully.")
