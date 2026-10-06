"""Train the TCT-IDS model reproducing Gong et al. (2026)."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import dagshub
import mlflow
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import CosineDecay

from data.tct_ids_dataset import TCTIDSDataset
from model.tct_ids import (
    build_tct_ids,
    enable_rdrop_training,
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
model_params = load_params("model.tct_ids")
training = load_params("training.tct_ids")
mlflow_params = load_params("mlflow")
window_size = int(split["window_size"])

best_path = "checkpoints/tct_ids_best.keras"
final_path = "checkpoints/tct_ids_final.keras"
run_id_path = "checkpoints/tct_ids_mlflow_run_id.txt"
best_metrics_path = "reports/metrics/tct_ids_best_training_metrics.json"
report_path = "reports/model_summaries/tct_ids_model_report.txt"


# ==========================================================
# Dataset
# ==========================================================

train = TCTIDSDataset(
    f"{dataset['processed_dir']}/train.parquet",
    window_size=window_size,
    d_model=model_params.get("d_model", 10),
    alpha=float(model_params.get("alpha_smooth", 1e-7)),
    batch_size=training["batch_size"],
    shuffle=True,
    random_seed=split["random_seed"],
    source_path=dataset["prepared_path"],
)
val = TCTIDSDataset(
    f"{dataset['processed_dir']}/val.parquet",
    window_size=window_size,
    d_model=model_params.get("d_model", 10),
    alpha=float(model_params.get("alpha_smooth", 1e-7)),
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

model = build_tct_ids(
    window_size=window_size,
    d_model=model_params.get("d_model", 10),
    num_heads=model_params.get("num_heads", 1),
    num_layers=model_params.get("num_layers", 3),
    dim_feedforward=model_params.get("dim_feedforward", 40),
    mlp_hidden_dim=model_params.get("mlp_hidden_dim", 40),
    tcn_filters=model_params.get("tcn_filters", 100),
    tcn_kernel_size=model_params.get("tcn_kernel_size", 2),
    tcn_dilations=model_params.get("tcn_dilations", [1, 2, 4]),
    use_weight_norm=bool(model_params.get("use_weight_norm", True)),
    dropout=model_params.get("dropout", 0.1),
    num_classes=num_classes,
    fusion=model_params.get("fusion", "concat"),
    pooling=model_params.get("pooling", "last"),
)

# Attach R-Drop regularized training step
lambda_rdrop = float(training.get("lambda_rdrop", 0.1))
enable_rdrop_training(model, lambda_rdrop=lambda_rdrop)

# Cosine decay schedule with linear warmup
steps_per_epoch = max(1, len(train.y) // training["batch_size"])
total_steps = steps_per_epoch * training["epochs"]
warmup_epochs = int(training.get("warmup_epochs", 50))
warmup_steps = warmup_epochs * steps_per_epoch

lr_schedule = CosineDecay(
    initial_learning_rate=float(training["learning_rate"]),
    decay_steps=total_steps,
    warmup_steps=min(warmup_steps, max(1, total_steps // 6)),
)

weight_decay = float(training.get("weight_decay", 1e-5))
optimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule,
    weight_decay=weight_decay if weight_decay > 0 else None,
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
    "tct_ids",
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
    mlflow_params.get("experiment_tct_ids", "can_ids_tct")
)
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

print("TCT-IDS training finished successfully.")
