"""Flower NumPyClient implementation for Federated CAN-IDS."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np

from fl.models import get_model_and_dataset


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

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        self.model.set_weights(parameters)

        tf_dataset = self.dataset.to_tf_dataset()
        history = self.model.fit(
            tf_dataset,
            epochs=self.local_epochs,
            verbose=0,
        )

        loss = float(history.history["loss"][-1]) if "loss" in history.history else 0.0
        acc = float(history.history["accuracy"][-1]) if "accuracy" in history.history else 0.0

        return self.model.get_weights(), len(self.dataset), {
            "loss": loss,
            "accuracy": acc,
        }

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        self.model.set_weights(parameters)
        tf_dataset = self.dataset.to_tf_dataset()
        results = self.model.evaluate(tf_dataset, verbose=0)
        return float(results[0]), len(self.dataset), {"accuracy": float(results[1])}
