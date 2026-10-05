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


def build_can_st_transformer(
    window_size: int = 64,
    vocab_size: int = 2306,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    dim_feedforward: int = 64,
    dropout: float = 0.0,
    batch_size: int | None = None,
) -> keras.Model:
    """Build the Spatial-Temporal CAN Transformer model using standard Keras layers.

    Architecture:
    1. Shared token and positional embeddings (Eq. 10-12)
    2. Temporal branch: Stack of Post-LN Transformer layers (MHA + FFN)
    3. Spatial branch: Stack of Post-LN Transformer layers with causal mask (Eq. 14-16)
    4. Representation extraction from last sequence slot
    5. Linear projection to vocab_size and vector addition (Eq. 24)
    """
    temp_in = keras.Input(
        shape=(window_size,), batch_size=batch_size, dtype="int32", name="temporal_input"
    )
    spat_in = keras.Input(
        shape=(8,), batch_size=batch_size, dtype="int32", name="spatial_input"
    )

    # 1. Embeddings (token embedding + broadcasted positional embedding)
    token_embed = layers.Embedding(vocab_size, d_model, name="token_embedding")
    pos_embed = layers.Embedding(256, d_model, name="pos_embedding")

    temp_pos = tf.range(window_size)[tf.newaxis, :]
    spat_pos = tf.range(8)[tf.newaxis, :]

    x_temp = token_embed(temp_in) + pos_embed(temp_pos)
    x_spat = token_embed(spat_in) + pos_embed(spat_pos)

    key_dim = max(1, d_model // num_heads)

    # 2. Temporal Transformer Branch (Post-LN)
    for i in range(num_layers):
        idx = i + 1
        attn_temp = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            name=f"temporal_block_{idx}_mha",
        )(x_temp, x_temp)
        res1 = x_temp + attn_temp
        norm1 = layers.LayerNormalization(
            epsilon=1e-5, name=f"temporal_block_{idx}_ln1"
        )(res1)

        ffn1 = layers.Dense(
            dim_feedforward, activation="relu", name=f"temporal_block_{idx}_ffn1"
        )(norm1)
        ffn2 = layers.Dense(d_model, name=f"temporal_block_{idx}_ffn2")(ffn1)
        res2 = norm1 + ffn2
        x_temp = layers.LayerNormalization(
            epsilon=1e-5, name=f"temporal_block_{idx}_ln2"
        )(res2)

    # 3. Spatial Transformer Branch (Post-LN, Causal Mask)
    for i in range(num_layers):
        idx = i + 1
        attn_spat = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            name=f"spatial_block_{idx}_mha",
        )(x_spat, x_spat, use_causal_mask=True)
        res1 = x_spat + attn_spat
        norm1 = layers.LayerNormalization(
            epsilon=1e-5, name=f"spatial_block_{idx}_ln1"
        )(res1)

        ffn1 = layers.Dense(
            dim_feedforward, activation="relu", name=f"spatial_block_{idx}_ffn1"
        )(norm1)
        ffn2 = layers.Dense(d_model, name=f"spatial_block_{idx}_ffn2")(ffn1)
        res2 = norm1 + ffn2
        x_spat = layers.LayerNormalization(
            epsilon=1e-5, name=f"spatial_block_{idx}_ln2"
        )(res2)

    # 4. Extract last token vectors
    cropped_temp = layers.Cropping1D(
        cropping=(window_size - 1, 0), name="temp_crop"
    )(x_temp)
    vec_temp = layers.Reshape((d_model,), name="last_temporal_vector")(cropped_temp)

    cropped_spat = layers.Cropping1D(
        cropping=(8 - 1, 0), name="spat_crop"
    )(x_spat)
    vec_spat = layers.Reshape((d_model,), name="last_spatial_vector")(cropped_spat)

    # 5. Output Projection Head & Vector Addition (Eq. 24)
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
