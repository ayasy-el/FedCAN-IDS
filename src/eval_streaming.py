"""Evaluate the causal model in chronological streaming chunks."""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from data.streaming_dataset import StreamingCANDataset
from model.streaming_ids import StreamingCANIDS
from utils.params import load_params

dataset = load_params("dataset")
model_cfg = load_params("model.streaming")
training = load_params("training.streaming")
test = StreamingCANDataset(
    f"{dataset['featured_dir']}/test.parquet",
    training["chunk_len"],
    training["batch_size"],
    False,
    "checkpoints/streaming_norm_stats.json",
    False,
)
model = StreamingCANIDS(**model_cfg)
model(
        {
            "can_id": tf.zeros((training["batch_size"], training["chunk_len"]), tf.int32),
            "numeric": tf.zeros((training["batch_size"], training["chunk_len"], 11), tf.float32),
            "stream_id": tf.fill((training["batch_size"],), b"build"),
            "valid_mask": tf.ones((training["batch_size"], training["chunk_len"]), tf.float32),
        }
)
model.load_weights("checkpoints/streaming_best.keras")
model.initialize_stream_state(training["batch_size"])
y_true, y_pred = [], []
for batch, labels in test.to_tf_dataset():
    model._reset_if_new_stream(batch["stream_id"])
    logits, state = model.run_sequence(
        batch["can_id"], batch["numeric"], model._persistent_state(),
        training=False, active_mask=batch["valid_mask"]
    )
    model._store_stream_state(state, batch["stream_id"])
    pred = tf.argmax(logits, axis=-1).numpy()
    mask = batch["valid_mask"].numpy().astype(bool)
    y_true.extend(labels.numpy()[mask].reshape(-1))
    y_pred.extend(pred[mask].reshape(-1))
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
result = {
    "model": "causal_compressed_kv",
    "n_test_frames": len(y_true),
    "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    "classification_report": report,
}
Path("reports/metrics").mkdir(parents=True, exist_ok=True)
Path("reports/metrics/streaming_classification_report.json").write_text(
    json.dumps(result, indent=2)
)
print(json.dumps(result, indent=2))
