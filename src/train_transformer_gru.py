"""Train the Lightweight Transformer-GRU model with Nibble Lookup Table."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import dagshub
import mlflow
from tensorflow import keras

from data.cnn_lstm_dataset import CNNLSTMDataset
from model.transformer_gru import (
    build_transformer_gru,
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
task = validate_experiment(params, "transformer_gru")
num_classes = task["num_classes"]
split = params["split"]
model_params = load_params("model.transformer_gru")
training = load_params("training.transformer_gru")
mlflow_params = load_params("mlflow")
window_size = int(split.get("window_size", 20))

best_path = "checkpoints/transformer_gru_best.keras"
final_path = "checkpoints/transformer_gru_final.keras"
run_id_path = "checkpoints/transformer_gru_mlflow_run_id.txt"
best_metrics_path = "reports/metrics/transformer_gru_best_training_metrics.json"
report_path = "reports/model_summaries/transformer_gru_model_report.txt"


# ==========================================================
# Dataset (Reuses CNNLSTMDataset directly)
# ==========================================================

train = CNNLSTMDataset(
    f"{dataset['processed_dir']}/train.parquet",
    window_size=window_size,
    rolling_w=model_params.get("rolling_w", 50),
    num_features=model_params.get("num_features", 5),
    batch_size=training["batch_size"],
    shuffle=True,
    random_seed=split.get("random_seed", 42),
    source_path=dataset["prepared_path"],
    label_rule=model_params.get("label_rule", "last_frame"),
)
val = CNNLSTMDataset(
    f"{dataset['processed_dir']}/val.parquet",
    window_size=window_size,
    rolling_w=model_params.get("rolling_w", 50),
    num_features=model_params.get("num_features", 5),
    batch_size=training["batch_size"],
    shuffle=False,
    random_seed=split.get("random_seed", 42),
    source_path=dataset["prepared_path"],
    label_rule=model_params.get("label_rule", "last_frame"),
)
has_validation = len(val.y) > 0
monitor = "val_f1_macro" if has_validation else "f1_macro"


# ==========================================================
# Model and build
# ==========================================================

model = build_transformer_gru(
    window_size=window_size,
    num_features=model_params.get("num_features", 5),
    num_classes=num_classes,
    d_model=model_params.get("d_model", 16),
    num_heads=model_params.get("num_heads", 2),
    dim_feedforward=model_params.get("dim_feedforward", 32),
    dropout=model_params.get("dropout", 0.2),
    gru_hidden_size=model_params.get("gru_hidden_size", 32),
)

optimizer = keras.optimizers.Adam(
    learning_rate=float(training["learning_rate"])
)

model.compile(
    optimizer=optimizer,
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
    "transformer_gru",
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
mlflow.set_experiment(
    mlflow_params.get("experiment_transformer_gru", "can_ids_transformer_gru")
)
with mlflow.start_run() as run:
    save_run_id(run.info.run_id, run_id_path)
    mlflow.log_params({f"model.{key}": value for key, value in model_params.items()})
    mlflow.log_params({f"training.{key}": value for key, value in training.items()})
    mlflow.log_params({f"split.{key}": value for key, value in split.items()})
    mlflow.log_params(parameter_counts(model))
    log_artifact_organized(report_path, "reports/model_summaries")

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

print("Transformer-GRU training finished successfully.")
