"""Train the causal single-frame model using chronological TBPTT chunks."""

import dagshub
import mlflow
import tensorflow as tf
from tensorflow import keras

from data.streaming_dataset import StreamingCANDataset
from model.streaming_ids import StreamingCANIDS
from utils.metrics import MacroF1Score
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
optimizer = keras.optimizers.Adam(training["learning_rate"])
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def detach_state(state):
    return tuple(tf.stop_gradient(x) for x in state)


def run_epoch(model, data, optimizer=None):
    """Run chunks chronologically while carrying state per session."""
    training_mode = optimizer is not None
    loss_mean = keras.metrics.Mean()
    accuracy = keras.metrics.SparseCategoricalAccuracy()
    macro_f1 = MacroF1Score(model_params["num_classes"])
    state, previous_stream = None, None

    for batch, labels in data:
        stream = batch.pop("stream_id").numpy()[0]
        if previous_stream != stream:
            state = None
            previous_stream = stream

        if training_mode:
            with tf.GradientTape() as tape:
                logits, next_state = model.run_sequence(
                    batch["can_id"], batch["numeric"], state, training=True
                )
                loss = loss_fn(labels, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            logits, next_state = model.run_sequence(
                batch["can_id"], batch["numeric"], state, training=False
            )
            loss = loss_fn(labels, logits)

        state = detach_state(next_state)
        loss_mean.update_state(loss)
        accuracy.update_state(labels, logits)
        macro_f1.update_state(labels, logits)

    return float(loss_mean.result()), float(accuracy.result()), float(macro_f1.result())


mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_streaming"])
with mlflow.start_run() as run:
    save_run_id(run.info.run_id, "checkpoints/streaming_mlflow_run_id.txt")
    mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
    mlflow.log_params({f"training.{k}": v for k, v in training.items()})
    best_f1 = -1.0
    stale_epochs = 0
    for epoch in range(training["epochs"]):
        train_metrics = run_epoch(model, train.to_tf_dataset(), optimizer)
        val_metrics = run_epoch(model, val.to_tf_dataset())
        print(
            f"Epoch {epoch + 1}: train_loss={train_metrics[0]:.4f} "
            f"val_loss={val_metrics[0]:.4f} val_macro_f1={val_metrics[2]:.4f}"
        )
        mlflow.log_metrics({
            "loss": train_metrics[0], "accuracy": train_metrics[1],
            "macro_f1": train_metrics[2], "val_loss": val_metrics[0],
            "val_accuracy": val_metrics[1], "val_macro_f1": val_metrics[2],
        }, step=epoch)
        if val_metrics[2] > best_f1:
            best_f1, stale_epochs = val_metrics[2], 0
            model.save("checkpoints/streaming_best.keras")
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break
    model.save("checkpoints/streaming_final.keras")
    mlflow.log_artifact("checkpoints/streaming_best.keras")
    mlflow.log_artifact("checkpoints/streaming_final.keras")
    mlflow.log_artifact(stats_path)
print("Streaming training finished.")
