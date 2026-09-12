"""Train the TensorFlow CAN-BiGRUBERT reproduction."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import json

import dagshub
import mlflow
from tensorflow import keras

from data.can_bigrubert_dataset import CANBiGRUBERTDataset
from model.can_bigrubert import build_can_bigrubert, parameter_counts
from utils.metrics import MacroF1Score, MacroPrecision, MacroRecall
from utils.mlflow_utils import (
    BestEpochMetrics,
    MlflowEpochLogger,
    log_artifact_organized,
    save_run_id,
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
model_params = params["model"]["can_bigrubert"]
training = params["training"]["can_bigrubert"]
mlflow_params = params["mlflow"]
window_size = int(split["window_size"])
best_path = "checkpoints/can_bigrubert_best.keras"
final_path = "checkpoints/can_bigrubert_final.keras"
run_id_path = "checkpoints/can_bigrubert_mlflow_run_id.txt"
best_metrics_path = "reports/metrics/can_bigrubert_best_training_metrics.json"


# ==========================================================
# Dataset
# ==========================================================

train = CANBiGRUBERTDataset(
    f"{dataset['processed_dir']}/train.parquet",
    model_params["tokenizer_checkpoint"],
    window_size,
    model_params["max_length"],
    training["batch_size"],
    True,
    split["random_seed"],
)
val = CANBiGRUBERTDataset(
    f"{dataset['processed_dir']}/val.parquet",
    model_params["tokenizer_checkpoint"],
    window_size,
    model_params["max_length"],
    training["batch_size"],
    False,
    split["random_seed"],
)


# ==========================================================
# Model and build
# ==========================================================

model = build_can_bigrubert(
    window_size,
    model_params["max_length"],
    model_params["bert_checkpoint"],
    model_params["bigru_hidden_size"],
    model_params["dropout"],
    num_classes,
)
model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        MacroPrecision(num_classes),
        MacroRecall(num_classes),
        MacroF1Score(num_classes),
    ],
)
model.summary()
counts = parameter_counts(model)
print(json.dumps(counts, indent=2))


# ==========================================================
# MLflow tracking
# ==========================================================

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_can_bigrubert"])
with mlflow.start_run() as run:
    save_run_id(run.info.run_id, run_id_path)
    mlflow.log_params({f"split.{key}": value for key, value in split.items()})
    mlflow.log_params({f"model.{key}": value for key, value in model_params.items()})
    mlflow.log_params({f"training.{key}": value for key, value in training.items()})
    mlflow.log_param("window_size", window_size)
    mlflow.log_param("stride", split["stride"])
    mlflow.log_metrics({key: float(value) for key, value in counts.items()})
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=training["early_stopping_patience"],
            min_delta=training["early_stopping_min_delta"],
            restore_best_weights=True,
        ),
        BestEpochMetrics(best_metrics_path, monitor="val_loss", mode="min"),
        MlflowEpochLogger(),
    ]
    model.fit(
        train.to_tf_dataset(),
        validation_data=val.to_tf_dataset(),
        epochs=training["epochs"],
        callbacks=callbacks,
    )
    model.save(final_path)
    log_artifact_organized(best_path)
    log_artifact_organized(final_path)
    log_artifact_organized(best_metrics_path)
print("CAN-BiGRUBERT training finished.")
