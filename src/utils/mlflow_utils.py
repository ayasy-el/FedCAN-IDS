"""
The module provides two main components:

1. `MlflowEpochLogger`
   A Keras callback that logs all epoch-level metrics and losses to the
   currently active MLflow run. Used by the training scripts.

2. `save_run_id` / `load_run_id`
   Persist and retrieve the MLflow run ID so evaluation scripts can resume
   the same run created during training. This keeps training metrics
   (e.g. loss and accuracy) and final evaluation metrics (e.g.
   per-class precision, recall, and F1) in a single MLflow run, making
   them easier to inspect and compare in the MLflow UI.
"""

from pathlib import Path

import mlflow
from tensorflow import keras


class MlflowEpochLogger(keras.callbacks.Callback):
    """
    Log all metrics reported by Keras at the end of each epoch.

    This includes standard metrics such as `loss`, `val_loss`, and
    `accuracy`, as well as any custom metrics configured in
    `model.compile()`.
    """

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return

        clean_logs = {k: float(v) for k, v in logs.items() if v is not None}

        mlflow.log_metrics(clean_logs, step=epoch)


def save_run_id(run_id: str, path: str) -> None:
    """Persist the active MLflow run ID for later use by evaluation scripts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(run_id)


def load_run_id(path: str) -> str | None:
    """
    Load the persisted MLflow run ID for use by evaluation scripts.

    Returns
    -------
    str or None
        The run ID if the file exists, otherwise None.
    """
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text().strip()
