import json
import os
from pathlib import Path

import mlflow
from tensorflow import keras


def is_kaggle() -> bool:
    """Check if running inside a Kaggle notebook or script environment."""
    if (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
        or "KAGGLE_DATA_PROXY_TOKEN" in os.environ
        or Path("/kaggle").exists()
    ):
        return True
    try:
        import kaggle_secrets  # noqa: F401

        return True
    except ImportError:
        return False


def setup_kaggle_dagshub_credentials() -> bool:
    """
    If running inside Kaggle environment, retrieve credentials from Kaggle Secrets
    and set environment variables for DagsHub and MLflow.
    """
    if not is_kaggle():
        return False

    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        dagshub_token = None
        for secret_name in ("DAGSHUB_TOKEN", "DAGSHUB_USER_TOKEN"):
            try:
                dagshub_token = user_secrets.get_secret(secret_name)
                if dagshub_token:
                    break
            except Exception:
                continue

        if dagshub_token:
            os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
            os.environ["DAGSHUB_TOKEN"] = dagshub_token
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
            try:
                import dagshub.common.config as dagshub_config

                dagshub_config.token = dagshub_token
            except Exception:
                pass
            return True
        else:
            print(
                "Warning: Running on Kaggle, but DAGSHUB_TOKEN was not found in Kaggle Secrets."
            )
    except Exception as e:
        print(f"Warning: Failed to retrieve DAGSHUB_TOKEN from Kaggle Secrets: {e}")
    return False


# Auto-detect Kaggle on import to configure secrets if available
setup_kaggle_dagshub_credentials()


def init_dagshub(
    repo_owner: str = "ayasy-el",
    repo_name: str = "FedCAN-IDS",
    mlflow: bool = True,
    **kwargs,
):
    """
    Initialize DagsHub with Kaggle secret support if in Kaggle,
    otherwise fallback to default DagsHub initialization.
    """
    setup_kaggle_dagshub_credentials()
    import dagshub

    return dagshub.init(
        repo_owner=repo_owner, repo_name=repo_name, mlflow=mlflow, **kwargs
    )


setup_dagshub = init_dagshub
setup_dagshub_credentials = setup_kaggle_dagshub_credentials



def write_model_report(
    path: str | Path,
    model_name: str,
    model: keras.Model,
    sections: dict,
    parameter_counts: dict | None = None,
) -> Path:
    """Write a reproducible text report for a trained model configuration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    counts = parameter_counts or {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(
            sum(int(weight.numpy().size) for weight in model.trainable_weights)
        ),
        "non_trainable_parameters": int(
            sum(int(weight.numpy().size) for weight in model.non_trainable_weights)
        ),
    }

    lines = [
        f"Model report: {model_name}",
        "=" * 80,
        "",
        "Configuration",
        "-------------",
        json.dumps(sections, indent=2, sort_keys=True, default=str),
        "",
        "Parameter counts",
        "----------------",
        json.dumps(counts, indent=2, sort_keys=True),
        "",
        "Keras model summary",
        "-------------------",
        *summary_lines,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def log_artifact_organized(path: str | Path, artifact_dir: str | None = None):
    """Log an artifact under a consistent MLflow folder."""
    path = Path(path)
    if artifact_dir is None:
        name = path.name
        if path.suffix == ".keras":
            artifact_dir = "models"
        elif "norm_stats" in name:
            artifact_dir = "preprocessing"
        elif "confusion_matrix" in name:
            artifact_dir = "figures/confusion_matrices"
        elif "classification_report" in name:
            artifact_dir = "reports"
        elif "best_training_metrics" in name:
            artifact_dir = "metrics"
        else:
            artifact_dir = "artifacts"
    mlflow.log_artifact(str(path), artifact_path=artifact_dir)


class MlflowEpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return

        clean_logs = {k: float(v) for k, v in logs.items() if v is not None}

        mlflow.log_metrics(clean_logs, step=epoch)


class BestEpochMetrics(keras.callbacks.Callback):
    def __init__(self, output_path: str, monitor="val_f1_macro", mode="max"):
        super().__init__()
        self.output_path = Path(output_path)
        self.monitor = monitor
        self.mode = mode
        self.best_value = None
        self.best_epoch = None
        self.best_metrics = None

    def _is_better(self, value):
        if self.best_value is None:
            return True
        return (
            value > self.best_value if self.mode == "max" else value < self.best_value
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None or not self._is_better(float(value)):
            return
        self.best_value = float(value)
        self.best_epoch = int(epoch + 1)
        self.best_metrics = {
            key: float(metric_value)
            for key, metric_value in logs.items()
            if metric_value is not None
        }

    def on_train_end(self, logs=None):
        if self.best_metrics is None:
            return
        report = {
            "selection": {"monitor": self.monitor, "mode": self.mode},
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(report, indent=2))
        if mlflow.active_run() is not None:
            for metric_name, metric_value in self.best_metrics.items():
                mlflow.log_metric(
                    f"{metric_name}_best",
                    metric_value,
                    step=self.best_epoch,
                )
            log_artifact_organized(self.output_path, "metrics")


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
