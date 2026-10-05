"""Train the Spatial-Temporal CAN Transformer (Jo & Kim, 2024)."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import dagshub
import mlflow
from tensorflow import keras

from data.can_st_transformer_dataset import CANSTTransformerDataset
from model.can_st_transformer import (
    build_can_st_transformer,
    parameter_counts,
)
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
split = params["split"]
model_params = load_params("model.can_st_transformer")
training = load_params("training.can_st_transformer")
mlflow_params = load_params("mlflow")

window_size = int(model_params.get("window_size", split.get("window_size", 64)))
vocab_size = int(model_params.get("vocab_size", 2306))
d_model = int(model_params.get("d_model", 64))
num_heads = int(model_params.get("num_heads", 4))
num_layers = int(model_params.get("num_layers", 4))
dim_feedforward = int(model_params.get("dim_feedforward", 64))
dropout = float(model_params.get("dropout", 0.0))

best_path = "checkpoints/can_st_transformer_best.keras"
final_path = "checkpoints/can_st_transformer_final.keras"
run_id_path = "checkpoints/can_st_transformer_mlflow_run_id.txt"
best_metrics_path = "reports/metrics/can_st_transformer_best_training_metrics.json"
report_path = "reports/model_summaries/can_st_transformer_model_report.txt"


# ==========================================================
# Dataset
# ==========================================================

train = CANSTTransformerDataset(
    f"{dataset['processed_dir']}/train.parquet",
    source_path=dataset["prepared_path"],
    window_size=window_size,
    batch_size=training["batch_size"],
    shuffle=True,
    random_seed=split.get("random_seed", 42),
)

val = CANSTTransformerDataset(
    f"{dataset['processed_dir']}/val.parquet",
    source_path=dataset["prepared_path"],
    window_size=window_size,
    batch_size=training["batch_size"],
    shuffle=False,
    random_seed=split.get("random_seed", 42),
)
has_validation = len(val) > 0
monitor = "val_accuracy" if has_validation else "accuracy"


# ==========================================================
# Model Build & Compile
# ==========================================================

model = build_can_st_transformer(
    window_size=window_size,
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=float(training["learning_rate"])),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        "accuracy",
        keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_acc"),
    ],
)
model.summary()

write_model_report(
    report_path,
    "can_st_transformer",
    model,
    {
        "dataset": dataset,
        "task": task,
        "split": split,
        "model": model_params,
        "training": training,
        "parameter_counts": parameter_counts(model),
        "window_size": window_size,
        "vocab_size": vocab_size,
    },
)


# ==========================================================
# MLflow Tracking & Training
# ==========================================================

mlflow.set_tracking_uri(mlflow_params["tracking_uri"]) 
mlflow.set_experiment(
    mlflow_params.get("experiment_can_st_transformer", "can_st_transformer")
)

with mlflow.start_run() as run:
    save_run_id(run.info.run_id, run_id_path)
    mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
    mlflow.log_params({f"training.{k}": v for k, v in training.items()})
    mlflow.log_params({f"split.{k}": v for k, v in split.items()})
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

print("Spatial-Temporal CAN Transformer training finished successfully.")
