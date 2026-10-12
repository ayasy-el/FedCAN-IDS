"""Train the Lightweight CNN-LSTM model reproducing Singh et al. (2026)."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import dagshub
import mlflow
from tensorflow import keras

from data.cnn_lstm_dataset import CNNLSTMDataset
from model.cnn_lstm import (
    build_cnn_lstm,
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
task = validate_experiment(params, "cnn_lstm")
num_classes = task["num_classes"]
split = params["split"]
model_params = load_params("model.cnn_lstm")
training = load_params("training.cnn_lstm")
mlflow_params = load_params("mlflow")
window_size = int(split.get("window_size", 20))

best_path = "checkpoints/cnn_lstm_best.keras"
final_path = "checkpoints/cnn_lstm_final.keras"
run_id_path = "checkpoints/cnn_lstm_mlflow_run_id.txt"
best_metrics_path = "reports/metrics/cnn_lstm_best_training_metrics.json"
report_path = "reports/model_summaries/cnn_lstm_model_report.txt"


# ==========================================================
# Dataset
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

model = build_cnn_lstm(
    window_size=window_size,
    num_features=model_params.get("num_features", 5),
    num_classes=num_classes,
    cnn_filters=model_params.get("cnn_filters", 16),
    cnn_kernel_size=model_params.get("cnn_kernel_size", 3),
    dropout=model_params.get("dropout", 0.2),
    lstm_hidden_size=model_params.get("lstm_hidden_size", 32),
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
    "cnn_lstm",
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
    mlflow_params.get("experiment_cnn_lstm", "can_ids_cnn_lstm")
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

print("CNN-LSTM training finished successfully.")
