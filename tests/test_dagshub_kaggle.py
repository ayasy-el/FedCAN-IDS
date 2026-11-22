import os
import sys
from unittest.mock import MagicMock, patch

from utils.mlflow_utils import (
    init_dagshub,
    is_kaggle,
    setup_kaggle_dagshub_credentials,
)


def test_is_kaggle_false_when_not_in_kaggle():
    with patch.dict(os.environ, {}, clear=True):
        with patch("pathlib.Path.exists", return_value=False):
            assert is_kaggle() is False


def test_is_kaggle_true_when_env_var_present():
    with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
        assert is_kaggle() is True

    with patch.dict(os.environ, {"KAGGLE_URL_BASE": "https://www.kaggle.com"}):
        assert is_kaggle() is True


def test_setup_kaggle_dagshub_credentials_non_kaggle():
    with patch.dict(os.environ, {}, clear=True):
        with patch("pathlib.Path.exists", return_value=False):
            result = setup_kaggle_dagshub_credentials()
            assert result is False
            assert "DAGSHUB_USER_TOKEN" not in os.environ


def test_setup_kaggle_dagshub_credentials_in_kaggle_with_secret():
    mock_secrets_client = MagicMock()
    mock_instance = mock_secrets_client.return_value
    mock_instance.get_secret.side_effect = lambda name: "fake_token_123" if name == "DAGSHUB_TOKEN" else None

    mock_module = MagicMock()
    mock_module.UserSecretsClient = mock_secrets_client

    with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, clear=True):
        with patch.dict(sys.modules, {"kaggle_secrets": mock_module}):
            result = setup_kaggle_dagshub_credentials()
            assert result is True
            assert os.environ.get("DAGSHUB_USER_TOKEN") == "fake_token_123"
            assert os.environ.get("DAGSHUB_TOKEN") == "fake_token_123"
            assert os.environ.get("MLFLOW_TRACKING_USERNAME") == "fake_token_123"
            assert os.environ.get("MLFLOW_TRACKING_PASSWORD") == "fake_token_123"


def test_init_dagshub_calls_dagshub_init_with_defaults():
    with patch("dagshub.init") as mock_init:
        init_dagshub()
        mock_init.assert_called_once_with(
            repo_owner="ayasy-el",
            repo_name="FedCAN-IDS",
            mlflow=True,
        )
