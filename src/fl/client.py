"""Flower NumPyClient implementation for Federated CAN-IDS."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
from tensorflow import keras

from fl.models import get_model_and_dataset
from utils.device_utils import configure_gpu_memory_growth

# Enable memory growth inside Ray worker process
configure_gpu_memory_growth()


class ClientEpochLogger(keras.callbacks.Callback):
    """Callback to log each local epoch for the client."""

    def __init__(self, client_id: int, round_num: Any, total_epochs: int):
        super().__init__()
        self.client_id = client_id
        self.round_num = round_num
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] | None = None):
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        acc = float(logs.get("accuracy", 0.0))
        f1 = logs.get("f1_macro")
        f1_text = f" | f1_macro={float(f1):.4f}" if f1 is not None else ""
        print(
            f"  [Round {self.round_num} | Client {self.client_id}] "
            f"Epoch {epoch + 1}/{self.total_epochs}: loss={loss:.4f} | acc={acc:.4f}{f1_text}",
            flush=True,
        )


class FedCanClient(fl.client.NumPyClient):
    """Flower client that trains a local CAN-IDS model on partitioned window data."""

    def __init__(
        self,
        client_id: int,
        model_name: str,
        client_path: str | Path,
        params: dict,
        local_epochs: int = 2,
        batch_size: int = 32,
    ):
        self.client_id = client_id
        self.model_name = model_name
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.params = params

        self.model, self.dataset = get_model_and_dataset(
            model_name=model_name,
            data_path=client_path,
            params=params,
            is_training=True,
            batch_size=batch_size,
        )

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return self.model.get_weights()

    def _get_num_samples(self) -> int:
        if hasattr(self.dataset, "__len__"):
            try:
                return len(self.dataset)
            except Exception:
                pass
        if hasattr(self.dataset, "y"):
            return len(self.dataset.y)
        if hasattr(self.dataset, "num_windows"):
            return int(self.dataset.num_windows)
        if hasattr(self.dataset, "index"):
            return len(self.dataset.index)
        return 0

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.model.set_weights(parameters)

        round_num = config.get("round", "?")
        logger_cb = ClientEpochLogger(self.client_id, round_num, self.local_epochs)

        tf_dataset = self.dataset.to_tf_dataset()
        history = self.model.fit(
            tf_dataset,
            epochs=self.local_epochs,
            verbose=0,
            callbacks=[logger_cb],
        )

        loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
        acc = float(history.history["accuracy"][-1]) if "accuracy" in history.history else 0.0

        metrics = {
            "client_id": self.client_id,
            "loss": loss,
            "accuracy": acc,
        }
        if "f1_macro" in history.history:
            metrics["f1_macro"] = float(history.history["f1_macro"][-1])

        return self.model.get_weights(), self._get_num_samples(), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        self.model.set_weights(parameters)
        tf_dataset = self.dataset.to_tf_dataset()
        results = self.model.evaluate(tf_dataset, verbose=0)
        return float(results[0]), self._get_num_samples(), {"accuracy": float(results[1])}
