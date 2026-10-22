"""TensorFlow/Keras implementation of Lightweight Transformer-GRU with Nibble Lookup Table for CAN Intrusion Detection.

Architecture Pipeline:
1. Payload Image Input: [B, W, 8, 8, 1] - 8x8 binary bit-matrix sequence
2. Temporal Features Input: [B, W, N_feat] - engineered CAN temporal features (default: 5)
3. Spatial Feature Extraction (per frame):
   - NibbleTokenizer: unpacks 8x8 bit-matrix into 16 nibbles (4-bit chunks, vocab size 16, values 0..15)
   - Nibble Embedding: Embedding(16, d_model) -> [16, d_model]
   - Positional Embedding: Embedding(16, d_model) for nibble positions 0..15
   - Transformer-Lite Block: MultiHeadAttention + LayerNorm + MLP (Dense-ReLU-Dense) + LayerNorm
   - Flatten -> 16 * d_model dimensions per frame (default: 16 * 16 = 256 dims)
4. Feature Fusion:
   - Concatenate along feature axis -> [B, W, 256 + N_feat] (261 with N_feat=5)
5. Recurrent Temporal Modeling:
   - GRU(units=32, return_sequences=False) -> [B, 32]
6. Classification Head:
   - Dense(num_classes, activation='softmax') -> [B, num_classes]
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="transformer_gru")
class NibbleTokenizer(layers.Layer):
    """Converts an 8x8 bit-matrix per CAN frame into 16 nibble integers (values 0..15).

    Each 8-byte CAN payload has 16 nibbles (4 bits each):
    - Upper 4 bits of Byte j -> Nibble 2*j
    - Lower 4 bits of Byte j -> Nibble 2*j + 1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Powers of 2 for a 4-bit binary vector: [8, 4, 2, 1]
        self.powers = tf.constant([8.0, 4.0, 2.0, 1.0], dtype=tf.float32)

    def call(self, inputs):
        # inputs shape: (..., 8, 8, 1) or (..., 8, 8)
        if inputs.shape[-1] == 1:
            x = tf.squeeze(inputs, axis=-1)
        else:
            x = inputs

        batch_dims = tf.shape(x)[:-2]
        # Ensure binary range [0.0, 1.0] for bits
        x = tf.clip_by_value(x, 0.0, 1.0)
        # Reshape the last two dimensions (8, 8) -> (16, 4)
        x_16x4 = tf.reshape(x, tf.concat([batch_dims, [16, 4]], axis=0))
        # Compute integer value for each nibble: sum(bits * [8, 4, 2, 1])
        nibble_values = tf.reduce_sum(x_16x4 * self.powers, axis=-1)
        nibble_ints = tf.cast(nibble_values, tf.int32)
        return tf.clip_by_value(nibble_ints, 0, 15)

    def compute_output_shape(self, input_shape):
        base_shape = (
            list(input_shape[:-3])
            if input_shape[-1] == 1
            else list(input_shape[:-2])
        )
        return tuple(base_shape + [16])

    def get_config(self):
        return super().get_config()


def build_transformer_gru(
    window_size: int = 20,
    num_features: int = 5,
    num_classes: int = 6,
    d_model: int = 16,
    num_heads: int = 2,
    dim_feedforward: int = 32,
    dropout: float = 0.2,
    gru_hidden_size: int = 32,
    batch_size: int | None = None,
) -> keras.Model:
    """Build multi-input Transformer-Lite (Nibble Lookup) + GRU model.

    Args:
        window_size: Number of CAN frames in sliding window (default 20).
        num_features: Number of temporal engineered features (default 5).
        num_classes: Number of classification classes (default 6 for ROAD masquerade).
        d_model: Hidden dimension for nibble tokens (default 16, giving 16*16=256 spatial dims).
        num_heads: Number of attention heads in Transformer-Lite (default 2).
        dim_feedforward: Hidden dimension of FFN in Transformer block (default 32).
        dropout: Dropout rate (default 0.2).
        gru_hidden_size: Hidden units in temporal GRU layer (default 32).
        batch_size: Optional fixed batch size.
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

    # -------------------------------------------------------------
    # 1. Spatial Payload Sub-Module (Per Frame via Transformer-Lite)
    # -------------------------------------------------------------
    frame_in = keras.Input(shape=(8, 8, 1), name="frame_input")

    # Step A: Convert 8x8 bit matrix to 16 nibbles [0..15]
    nibbles = NibbleTokenizer(name="nibble_tokenizer")(frame_in)  # (16,)

    # Step B: Nibble Categorical Embedding + Positional Embedding
    nibble_embed = layers.Embedding(
        input_dim=16,
        output_dim=d_model,
        name="nibble_embedding",
    )(nibbles)  # (16, d_model)

    pos_indices = tf.range(16)
    pos_embed = layers.Embedding(
        input_dim=16,
        output_dim=d_model,
        name="nibble_pos_embedding",
    )(pos_indices)  # (16, d_model)

    x = nibble_embed + pos_embed

    # Step C: Transformer-Lite Encoder Block
    key_dim = max(1, d_model // num_heads)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name="mha",
    )(x, x)
    attn_out = layers.Dropout(dropout, name="mha_dropout")(attn_out)
    x = layers.LayerNormalization(epsilon=1e-5, name="ln1")(x + attn_out)

    ffn = layers.Dense(dim_feedforward, activation="relu", name="ffn_dense1")(x)
    ffn = layers.Dropout(dropout, name="ffn_dropout")(ffn)
    ffn = layers.Dense(d_model, name="ffn_dense2")(ffn)
    x = layers.LayerNormalization(epsilon=1e-5, name="ln2")(x + ffn)

    # Step D: Flatten 16 tokens -> (16 * d_model) dimensions per frame
    frame_spatial = layers.Flatten(name="spatial_flatten")(x)

    frame_module = keras.Model(
        inputs=frame_in,
        outputs=frame_spatial,
        name="nibble_transformer_submodule",
    )

    # Apply spatial extractor across all timesteps in window
    spatial_seq = layers.TimeDistributed(
        frame_module, name="time_distributed_transformer"
    )(image_input)

    # -------------------------------------------------------------
    # 2. Feature Fusion
    # -------------------------------------------------------------
    fused_seq = layers.Concatenate(axis=-1, name="feature_fusion")(
        [spatial_seq, features_input]
    )

    # -------------------------------------------------------------
    # 3. Recurrent Temporal Modeling (GRU)
    # -------------------------------------------------------------
    gru_repr = layers.GRU(
        units=int(gru_hidden_size),
        return_sequences=False,
        name="gru",
    )(fused_seq)

    # -------------------------------------------------------------
    # 4. Classification Head
    # -------------------------------------------------------------
    outputs = layers.Dense(
        units=int(num_classes),
        activation="softmax",
        name="classification_head",
    )(gru_repr)

    return keras.Model(
        inputs=[image_input, features_input],
        outputs=outputs,
        name="transformer_gru",
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
