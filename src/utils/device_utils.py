"""Device and distributed training utilities (TPU, GPU, CPU)."""

import os
import tensorflow as tf


def is_tpu_available(tpu_name: str | None = None) -> bool:
    """Check if a TPU is available in the current environment."""
    if "COLAB_TPU_ADDR" in os.environ or "TPU_NAME" in os.environ:
        return True
    try:
        tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        return True
    except (ValueError, Exception):
        return False


def setup_device_strategy(tpu_name: str | None = None) -> tf.distribute.Strategy:
    """
    Detect if a TPU is available and initialize a TPUStrategy.
    If no TPU is detected, fallback to GPU (MirroredStrategy/default) or CPU.
    """
    # 1. Try modern TPU connect (Colab / Kaggle TPU VM / Cloud TPU)
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu_name)
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"[TPU] Successfully connected and initialized TPUStrategy on: {tpu.master()}")
        return strategy
    except Exception:
        pass

    # 2. Try explicit resolver connect + initialize_tpu_system
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"[TPU] Successfully initialized TPUStrategy on: {tpu.master()}")
        return strategy
    except Exception:
        pass

    # 3. Fallback to GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[GPU] Found {len(gpus)} GPU(s)")
        if len(gpus) > 1:
            return tf.distribute.MirroredStrategy()
        return tf.distribute.get_strategy()

    # 4. Fallback to CPU default strategy
    print("[CPU] No TPU or GPU detected, running on default CPU strategy")
    return tf.distribute.get_strategy()


get_strategy = setup_device_strategy
init_tpu_strategy = setup_device_strategy
