import mlflow
import tensorflow as tf
from tensorflow import keras

from data.dataset import CANDataset
from model.spatial_transformer import SpatialTransformer
from utils.mlflow_utils import MlflowEpochLogger, save_run_id
from utils.params import load_params

# ==========================
# Load params.yaml (DVC-tracked hyperparameter, bukan hardcode lagi)
# ==========================

dataset_params = load_params("dataset")
model_params = load_params("model.spatial")
training_params = load_params("training.spatial")
mlflow_params = load_params("mlflow")


# ==========================
# Configuration
# ==========================

TRAIN_PATH = f"{dataset_params['processed_dir']}/train.parquet"
VAL_PATH = f"{dataset_params['processed_dir']}/val.parquet"

BATCH_SIZE = training_params["batch_size"]
EPOCHS = training_params["epochs"]
LEARNING_RATE = training_params["learning_rate"]

RUN_ID_PATH = "checkpoints/spatial_mlflow_run_id.txt"


# ==========================
# MLflow setup
# ==========================

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_spatial"])


# ==========================
# Dataset
# ==========================

train_dataset = CANDataset(
    TRAIN_PATH,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_dataset = CANDataset(
    VAL_PATH,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

train_tf = train_dataset.to_tf_dataset()
val_tf = val_dataset.to_tf_dataset()


# ==========================
# Model
# ==========================

model = SpatialTransformer(**model_params)

# build model
dummy = {
    "tokens": tf.zeros((1, 10), dtype=tf.int32),
    "token_types": tf.zeros((1, 10), dtype=tf.int32),
    "positions": tf.zeros((1, 10), dtype=tf.int32),
}
model(dummy)
model.summary()


# ==========================
# Compile
# ==========================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)


# ==========================
# Callback
# ==========================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/spatial_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
    ),
]


# ==========================
# Training (dibungkus 1 MLflow run)
# ==========================

with mlflow.start_run() as run:
    # Simpan run_id supaya eval_spatial.py bisa log metric test ke run
    # MLflow yang SAMA (bukan run terpisah) -- gampang dibandingkan di UI.
    save_run_id(run.info.run_id, RUN_ID_PATH)

    mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
    mlflow.log_params({f"training.{k}": v for k, v in training_params.items()})

    callbacks.append(MlflowEpochLogger())

    history = model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    mlflow.log_artifact("checkpoints/spatial_best.keras")

print("Training selesai.")
