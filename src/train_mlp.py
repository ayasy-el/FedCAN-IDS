"""Train the frame-level MLP baseline."""

import dagshub
import mlflow
from tensorflow import keras

from data.mlp_dataset import MLPCANDataset
from model.mlp import build_mlp
from utils.metrics import MacroF1Score, MacroPrecision, MacroRecall
from utils.mlflow_utils import (
    BestEpochMetrics,
    MlflowEpochLogger,
    log_artifact_organized,
    save_run_id,
)
from utils.params import load_params

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)


dataset = load_params("dataset")
model_params = load_params("model.mlp")
training = load_params("training.mlp")
mlflow_params = load_params("mlflow")
stats_path = "checkpoints/mlp_norm_stats.json"
best_metrics_path = "reports/metrics/mlp_best_training_metrics.json"

train = MLPCANDataset(
    f"{dataset['featured_dir']}/train.parquet",
    stats_path,
    True,
    model_params["can_id_bits"],
    training["batch_size"],
    True,
)
val = MLPCANDataset(
    f"{dataset['featured_dir']}/val.parquet",
    stats_path,
    False,
    model_params["can_id_bits"],
    training["batch_size"],
)
model = build_mlp(train.input_dim, model_params["num_classes"])
model.compile(
    optimizer=keras.optimizers.Adam(training["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        MacroPrecision(model_params["num_classes"]),
        MacroRecall(model_params["num_classes"]),
        MacroF1Score(model_params["num_classes"]),
    ],
)
model.summary()
mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_mlp"])
with mlflow.start_run() as run:
    save_run_id(run.info.run_id, "checkpoints/mlp_mlflow_run_id.txt")
    mlflow.log_params({f"model.{key}": value for key, value in model_params.items()})
    mlflow.log_params({f"training.{key}": value for key, value in training.items()})
    model.fit(
        train.to_tf_dataset(),
        validation_data=val.to_tf_dataset(),
        epochs=training["epochs"],
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "checkpoints/mlp_best.keras",
                monitor="val_f1_macro",
                mode="max",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_f1_macro",
                mode="max",
                patience=5,
                restore_best_weights=True,
            ),
            BestEpochMetrics(best_metrics_path, monitor="val_f1_macro", mode="max"),
            MlflowEpochLogger(),
        ],
    )
    model.save("checkpoints/mlp_final.keras")
    log_artifact_organized("checkpoints/mlp_best.keras")
    log_artifact_organized("checkpoints/mlp_final.keras")
    log_artifact_organized(stats_path)
    log_artifact_organized(best_metrics_path)
print("MLP training finished.")
