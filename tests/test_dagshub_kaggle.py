import os
import sys
from unittest.mock import MagicMock, patch

from utils.device_utils import is_tpu_available, setup_device_strategy
from utils.mlflow_utils import (
    init_dagshub,
    is_colab,
    is_kaggle,
    setup_cloud_credentials,
    setup_colab_dagshub_credentials,
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


def test_is_colab_detection():
    with patch.dict(os.environ, {}, clear=True):
        assert is_colab() is False

    with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "1"}):
        assert is_colab() is True

    with patch.dict(os.environ, {"COLAB_GPU": "1"}):
        assert is_colab() is True


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


def test_setup_colab_dagshub_credentials_in_colab_with_userdata():
    mock_userdata = MagicMock()
    mock_userdata.get.side_effect = lambda name: "colab_token_456" if name == "DAGSHUB_TOKEN" else None

    mock_colab = MagicMock()
    mock_colab.userdata = mock_userdata

    with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "release-1"}, clear=True):
        with patch.dict(sys.modules, {"google.colab": mock_colab, "google.colab.userdata": mock_userdata}):
            result = setup_colab_dagshub_credentials()
            assert result is True
            assert os.environ.get("DAGSHUB_USER_TOKEN") == "colab_token_456"
            assert os.environ.get("DAGSHUB_TOKEN") == "colab_token_456"
            assert os.environ.get("MLFLOW_TRACKING_USERNAME") == "colab_token_456"
            assert os.environ.get("MLFLOW_TRACKING_PASSWORD") == "colab_token_456"


def test_setup_cloud_credentials_local_fallback():
    with patch.dict(os.environ, {}, clear=True):
        with patch("pathlib.Path.exists", return_value=False):
            result = setup_cloud_credentials()
            assert result is False
            assert "DAGSHUB_USER_TOKEN" not in os.environ


def test_init_dagshub_calls_dagshub_init_with_defaults():
    with patch("dagshub.init") as mock_init:
        init_dagshub()
        mock_init.assert_called_once_with(
            repo_owner="ayasy-el",
            repo_name="FedCAN-IDS",
            mlflow=True,
        )


def test_tpu_detection_and_strategy_fallback_when_no_tpu():
    with patch.dict(os.environ, {}, clear=True):
        strategy = setup_device_strategy()
        assert strategy is not None


def test_tpu_detection_when_tpu_mocked():
    import tensorflow as tf

    mock_tpu = MagicMock()
    mock_tpu.master.return_value = "grpc://10.0.0.2:8470"
    mock_strategy = MagicMock()

    with patch.object(tf.distribute.cluster_resolver.TPUClusterResolver, "connect", return_value=mock_tpu):
        with patch.object(tf.distribute, "TPUStrategy", return_value=mock_strategy):
            strategy = setup_device_strategy()
            assert strategy == mock_strategy
