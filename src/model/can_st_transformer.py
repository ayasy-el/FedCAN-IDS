"""TensorFlow/Keras reproduction of Spatial-Temporal CAN Transformer (Jo & Kim, 2024).

Reference:
Jo and Kim (2024), "Intrusion Detection Using Transformer in Controller Area Network",
IEEE Access, vol. 12, pp. 121932-121946.
"""

import os
from typing import Dict

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LearnablePositionalEmbedding(layers.Layer):
    """Learnable 1D positional embedding layer (Eq. 11-12)."""

    def __init__(self, max_len: int = 256, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(self.max_len, self.d_model),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_emb[:seq_len]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class ExtractLastToken(layers.Layer):
    """Extract representation from the last sequence position."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x[:, -1, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class TransformerBlock(layers.Layer):
    """Single Post-LN Transformer block reproducing Jo & Kim (2024).

    Structure:
    1. Multi-head self-attention -> Residual Add -> LayerNorm (Eq. 17)
    2. Feed-Forward Network (Linear -> ReLU -> Linear) -> Residual Add -> LayerNorm (Eq. 18-19)
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, d_model // num_heads),
            dropout=dropout,
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn_dense1 = layers.Dense(dim_feedforward, activation="relu", name="ffn_dense1")
        self.ffn_dense2 = layers.Dense(d_model, name="ffn_dense2")
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, use_causal_mask: bool = False):
        attn_out = self.mha(x, x, use_causal_mask=use_causal_mask)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn_dense2(self.ffn_dense1(x))
        x = self.ln2(x + ffn_out)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
            }
        )
        return config


def build_can_st_transformer(
    window_size: int = 64,
    vocab_size: int = 2306,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 64,
    dropout: float = 0.0,
    max_len: int = 256,
    batch_size: int | None = None,
) -> keras.Model:
    """Build the Spatial-Temporal CAN Transformer model.

    Inputs:
    - temporal_input: (window_size,) int32 tokens [t_{n-w+1}, ..., t_{n-1}, <Mask>]
    - spatial_input: (8,) int32 tokens [s0, ..., s7]

    Outputs:
    - combined_logits: (vocab_size,) float32 vector sum: v = v_temp + v_spat
    """
    temp_in = keras.Input(
        shape=(window_size,), batch_size=batch_size, dtype="int32", name="temporal_input"
    )
    spat_in = keras.Input(
        shape=(8,), batch_size=batch_size, dtype="int32", name="spatial_input"
    )

    token_embed = layers.Embedding(vocab_size, d_model, name="token_embedding")
    pos_embed = LearnablePositionalEmbedding(max_len=max_len, d_model=d_model, name="pos_embedding")

    x_temp = pos_embed(token_embed(temp_in))
    x_spat = pos_embed(token_embed(spat_in))

    for i in range(num_layers):
        block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            name=f"transformer_block_{i}",
        )
        x_temp = block(x_temp, use_causal_mask=False)
        x_spat = block(x_spat, use_causal_mask=True)

    # Extract representation from the last slot of each sequence using dedicated layer (safe for serialization)
    vec_temp = ExtractLastToken(name="last_temporal_vector")(x_temp)
    vec_spat = ExtractLastToken(name="last_spatial_vector")(x_spat)

    head = layers.Dense(vocab_size, name="logits_head")
    logits_temp = head(vec_temp)
    logits_spat = head(vec_spat)

    combined_logits = layers.Add(name="combined_logits")([logits_temp, logits_spat])

    return keras.Model(
        inputs=[temp_in, spat_in],
        outputs=combined_logits,
        name="can_st_transformer",
    )


def parameter_counts(model: keras.Model) -> Dict[str, int]:
    """Return trainable, non-trainable, and total parameter counts."""
    trainable = int(
        sum(tf.reduce_prod(w.shape).numpy() for w in model.trainable_weights)
    )
    non_trainable = int(
        sum(tf.reduce_prod(w.shape).numpy() for w in model.non_trainable_weights)
    )
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }
