"""TensorFlow/Keras implementation of Lightweight CNN-LSTM for CAN Intrusion Detection.

Reference:
Singh et al. (2026), "Efficient Intrusion Detection for Intra-Vehicle CAN Network
With Light-Weight Deep Learning-Based Model", IEEE Access, vol. 14, pp. 50811-50830.
DOI: 10.1109/ACCESS.2026.3675250
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras

layers = keras.layers


def build_cnn_lstm(
    window_size: int = 20,
    num_features: int = 5,
    num_classes: int = 6,
    cnn_filters: int = 16,
    cnn_kernel_size: int = 3,
    cnn_padding: str = "same",
    pool_size: tuple[int, int] = (2, 2),
    dropout: float = 0.2,
    lstm_hidden_size: int = 32,
    batch_size: int | None = None,
) -> keras.Model:
    """Build multi-input Lightweight CNN-LSTM model reproducing Singh et al. (2026).

    Architecture Pipeline (Fig. 5 & Section V-A4):
    1. Payload Image Input: [B, S, 8, 8, 1] - 8x8 binary bit-matrix sequence
    2. Temporal Features Input: [B, S, N_feat] - engineered CAN temporal features
    3. CNN Branch (per time-step):
       - Conv2D(16, (3, 3), padding='same') + ReLU
       - MaxPool2D((2, 2))
       - Dropout(0.2)
       - Flatten -> 256 dimensions per time step
    4. Feature Fusion:
       - Concatenation along feature axis -> [B, S, 256 + N_feat] (261 with N_feat=5)
    5. LSTM Branch:
       - LSTM(hidden_size=32, return_sequences=False) -> [B, 32] (last hidden state h^(S))
    6. Classification Head:
       - Dense(num_classes, activation='softmax') -> [B, num_classes]
    """
    image_input = keras.Input(
        shape=(window_size, 8, 8, 1),
        batch_size=batch_size,
        dtype=tf.float32,
        name="image_input",
    )
    features_input = keras.Input(
        shape=(window_size, num_features),
        batch_size=batch_size,
        dtype=tf.float32,
        name="features_input",
    )

    # Spatial Feature Extraction (per time step)
    cnn_module = keras.Sequential(
        [
            layers.Conv2D(
                filters=int(cnn_filters),
                kernel_size=(int(cnn_kernel_size), int(cnn_kernel_size)),
                padding=cnn_padding,
                activation="relu",
                name="conv2d",
            ),
            layers.MaxPooling2D(pool_size=pool_size, name="maxpool2d"),
            layers.Dropout(rate=float(dropout), name="dropout"),
            layers.Flatten(name="flatten"),
        ],
        name="cnn_submodule",
    )
    spatial_seq = layers.TimeDistributed(
        cnn_module, name="time_distributed_cnn"
    )(image_input)

    # Feature Fusion: Concatenate spatial features and engineered temporal features per time step
    fused_seq = layers.Concatenate(axis=-1, name="feature_fusion")(
        [spatial_seq, features_input]
    )

    # Recurrent Temporal Modeling: Extract last hidden state h^(S)
    lstm_repr = layers.LSTM(
        units=int(lstm_hidden_size),
        return_sequences=False,
        name="lstm",
    )(fused_seq)

    # Classification Head
    outputs = layers.Dense(
        units=int(num_classes),
        activation="softmax",
        name="classification_head",
    )(lstm_repr)

    return keras.Model(
        inputs=[image_input, features_input],
        outputs=outputs,
        name="cnn_lstm",
    )


def parameter_counts(model: keras.Model) -> dict[str, int]:
    """Return model parameter count metrics."""
    trainable_params = int(
        sum(tf.size(w).numpy() for w in model.trainable_weights)
    )
    non_trainable_params = int(
        sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    )
    total_params = trainable_params + non_trainable_params
    return {
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "total_parameters": total_params,
    }
