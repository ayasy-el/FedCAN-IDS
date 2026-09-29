"""TensorFlow/Keras reproduction of CAN-AE-Transformer-IDS.

Reference:
Le et al. (2024), "Multi-classification in-vehicle intrusion detection system
using packet- and sequence-level characteristics from time-embedded transformer
with autoencoder", Knowledge-Based Systems 299, 112091.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GlobalSumPooling1D(layers.Layer):
    """Global sum-pooling across temporal sequence dimension."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def build_can_ae_transformer(
    window_size: int = 15,
    d_model: int = 10,
    num_heads: int = 10,
    num_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    num_classes: int = 7,
    batch_size: int | None = None,
) -> keras.Model:
    """Build the CAN-AE-Transformer model reproducing Le et al. (2024).

    Architecture:
    1. Single message extraction (Autoencoder Encoder):
       Linear(12 -> 32) -> ReLU -> Linear(32 -> 16) -> ReLU -> Linear(16 -> d_model) -> ReLU
    2. Time-Positional Encoding (TSE):
       Precomputed sinusoidal time embedding (window_size, d_model) added element-wise
    3. Sequence feature extraction:
       Stack of num_layers Transformer Encoder layers (MHA + FFN with LayerNorm post-LN)
    4. Global Sum-Pooling over window sequence
    5. Linear Classifier:
       Linear(d_model -> num_classes) with Softmax
    """
    message_input = keras.Input(
        shape=(window_size, 12),
        batch_size=batch_size,
        dtype=tf.float32,
        name="message_features",
    )
    time_input = keras.Input(
        shape=(window_size, d_model),
        batch_size=batch_size,
        dtype=tf.float32,
        name="time_embeddings",
    )

    # 1. Single message extraction (Autoencoder Encoder)
    ae = layers.Dense(32, activation="relu", name="ae_dense_1")(message_input)
    ae = layers.Dense(16, activation="relu", name="ae_dense_2")(ae)
    ae_out = layers.Dense(d_model, activation="relu", name="ae_dense_3")(ae)

    # 2. Time-Positional Encoding injection
    x = layers.Add(name="tse_add")([ae_out, time_input])

    # 3. Transformer Encoder stack
    key_dim = max(1, d_model // num_heads)
    for i in range(num_layers):
        layer_idx = i + 1

        # Multi-Head Attention Sub-layer (Post-LN)
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            name=f"transformer_layer_{layer_idx}_mha",
        )(x, x)
        attn_out = layers.Dropout(
            dropout, name=f"transformer_layer_{layer_idx}_attn_dropout"
        )(attn_out)
        res_attn = layers.Add(name=f"transformer_layer_{layer_idx}_attn_res")([x, attn_out])
        x = layers.LayerNormalization(
            epsilon=1e-5, name=f"transformer_layer_{layer_idx}_attn_norm"
        )(res_attn)

        # Feed-Forward Network Sub-layer (Post-LN)
        ffn = layers.Dense(
            dim_feedforward,
            activation="relu",
            name=f"transformer_layer_{layer_idx}_ffn_1",
        )(x)
        ffn = layers.Dropout(
            dropout, name=f"transformer_layer_{layer_idx}_ffn_dropout_1"
        )(ffn)
        ffn = layers.Dense(
            d_model, name=f"transformer_layer_{layer_idx}_ffn_2"
        )(ffn)
        ffn = layers.Dropout(
            dropout, name=f"transformer_layer_{layer_idx}_ffn_dropout_2"
        )(ffn)
        res_ffn = layers.Add(name=f"transformer_layer_{layer_idx}_ffn_res")([x, ffn])
        x = layers.LayerNormalization(
            epsilon=1e-5, name=f"transformer_layer_{layer_idx}_ffn_norm"
        )(res_ffn)

    # 4. Global Sum-Pooling over window dimension
    pooled = GlobalSumPooling1D(name="sum_pooling")(x)

    # 5. Classification Head
    output = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_output",
    )(pooled)

    return keras.Model(
        inputs={
            "message_features": message_input,
            "time_embeddings": time_input,
        },
        outputs=output,
        name="can_ae_transformer",
    )


def parameter_counts(model: keras.Model) -> dict[str, int]:
    """Breakdown of trainable and total parameters by component."""
    def count(weights):
        return int(sum(int(parameter.numpy().size) for parameter in weights))

    ae_layers = [
        layer for layer in model.layers if layer.name.startswith("ae_dense")
    ]
    transformer_layers = [
        layer for layer in model.layers if layer.name.startswith("transformer_layer")
    ]
    classifier_layers = [
        layer for layer in model.layers if layer.name.startswith("classification_")
    ]

    ae_params = count([w for layer in ae_layers for w in layer.weights])
    transformer_params = count(
        [w for layer in transformer_layers for w in layer.weights]
    )
    classifier_params = count(
        [w for layer in classifier_layers for w in layer.weights]
    )

    return {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": count(model.trainable_weights),
        "non_trainable_parameters": count(model.non_trainable_weights),
        "ae_parameters": ae_params,
        "transformer_parameters": transformer_params,
        "classifier_parameters": classifier_params,
    }
