"""Train the causal single-frame model using chronological TBPTT chunks."""

import dagshub
import mlflow
import tensorflow as tf
from tensorflow import keras

from data.streaming_dataset import StreamingCANDataset
from model.streaming_ids import StreamingCANIDS
from utils.mlflow_utils import save_run_id
from utils.params import load_params

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)

dataset = load_params("dataset")
model_params = load_params("model.streaming")
training = load_params("training.streaming")
mlflow_params = load_params("mlflow")
train_path = f"{dataset['featured_dir']}/train.parquet"
val_path = f"{dataset['featured_dir']}/val.parquet"
stats_path = "checkpoints/streaming_norm_stats.json"

train = StreamingCANDataset(
    train_path, training["chunk_len"], training["batch_size"], False, stats_path, True
)
val = StreamingCANDataset(
    val_path, training["chunk_len"], training["batch_size"], False, stats_path, False
)

model = StreamingCANIDS(**model_params)
model(
    {
        "can_id": tf.zeros((1, training["chunk_len"]), tf.int32),
        "numeric": tf.zeros((1, training["chunk_len"], 11), tf.float32),
        "stream_id": tf.constant([b"build"]),
    }
)
model.summary()
model.initialize_stream_state(batch_size=training["batch_size"])
# Persistent streaming state is a resource variable.  Disable Keras' automatic
# XLA compilation so that the state can safely be read/written across devices.
# TensorFlow ops still execute on the GPU where supported.
model.compile(
    optimizer=keras.optimizers.Adam(training["learning_rate"]),
    jit_compile=False,
)


class ResetStreamingState(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.model.reset_stream_state()

    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_stream_state()

    def on_test_begin(self, logs=None):
        self.model.reset_stream_state()


callbacks = [
    keras.callbacks.ModelCheckpoint(
        "checkpoints/streaming_best.keras",
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_macro_f1", mode="max", patience=5, restore_best_weights=True
    ),
    ResetStreamingState(),
]


mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_streaming"])
with mlflow.start_run() as run:
    save_run_id(run.info.run_id, "checkpoints/streaming_mlflow_run_id.txt")
    mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
    mlflow.log_params({f"training.{k}": v for k, v in training.items()})
    history = model.fit(
        train.to_tf_dataset(),
        validation_data=val.to_tf_dataset(),
        epochs=training["epochs"],
        steps_per_epoch=train.num_samples,
        validation_steps=val.num_samples,
        shuffle=False,
        callbacks=callbacks,
    )
    for epoch, values in enumerate(zip(history.history["loss"], history.history["val_loss"])):
        mlflow.log_metrics({"loss": values[0], "val_loss": values[1]}, step=epoch)
    model.save("checkpoints/streaming_final.keras")
    mlflow.log_artifact("checkpoints/streaming_best.keras")
    mlflow.log_artifact("checkpoints/streaming_final.keras")
    mlflow.log_artifact(stats_path)
print("Streaming training finished.")
