"""TensorFlow/Keras implementation of HAT-IDS (Hierarchical Attention Transformer Intrusion Detection System).

Architecture Pipeline (Pure Attention - No GRU):
1. Frame Nibbles Input: [B, W, 19] (int32) - 19 nibbles per frame in range [0..15]:
   - nibbles 0..2: 3 CAN Arbitration ID nibbles (12 bits)
   - nibbles 3..18: 16 payload nibbles (upper & lower 4 bits for Data_0..Data_7)
2. Temporal Features Input: [B, W, N_feat] (float32) - 5 engineered CAN temporal features:
   (can_id_norm, time_delta, freq_id, bus_load, time_st)
3. Time Embeddings Input: [B, W, d_model] (float32) - Relative Time-Positional Encoding (TSE)
4. Level 1: Intra-Frame Spatial Transformer Submodule (per frame):
   - Shared Nibble Lookup Table: Embedding(16, d_model) -> [19, d_model]
   - Trainable [CLS] token prepended at index 0 -> [20, d_model]
   - Positional Embedding for tokens 0..19 -> [20, d_model]
   - MultiHeadAttention(num_heads, key_dim) + LayerNorm + MLP + LayerNorm
   - Frame Summary: Extracted [CLS] token at index 0 -> [d_model]
   - Output across window: [B, W, d_model]
5. Multi-Scale Linear Feature Fusion:
   - Concatenate([spatial_seq, features_input]) -> [B, W, d_model + N_feat]
   - Dense(d_model, activation='relu') -> [B, W, d_model]
6. Time-Positional Encoding (TSE) Injection:
   - Fused + Time_Embeddings -> [B, W, d_model]
   - LayerNormalization
7. Level 2: Inter-Frame Temporal Transformer Encoder:
   - MultiHeadAttention across full sliding window timesteps W -> [B, W, d_model]
   - Residual connection + LayerNorm + MLP (Dense-ReLU-Dense) + LayerNorm
8. Sequence Aggregation:
   - AttentionPooling1D across sequence dimension W -> [B, d_model]
9. Classification Head:
   - Dense(num_classes, activation='softmax') -> [B, num_classes]
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="hat_ids")
class CLSToken(layers.Layer):
    """Prepends a trainable [CLS] token to the sequence."""

    def __init__(self, d_model: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token_weight",
            shape=(1, 1, self.d_model),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcast = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_broadcast, inputs], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, self.d_model)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


@keras.utils.register_keras_serializable(package="hat_ids")
class ExtractCLS(layers.Layer):
    """Extracts the [CLS] token representation at index 0."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()


@keras.utils.register_keras_serializable(package="hat_ids")
class AttentionPooling1D(layers.Layer):
    """Learned context-aware attention pooling across sequence dimension."""

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.score_dense = layers.Dense(
            self.units, activation="tanh", name="score_tanh"
        )
        self.context_vector = layers.Dense(1, use_bias=False, name="context_vector")

    def call(self, inputs):
        scores = self.score_dense(inputs)
        weights = self.context_vector(scores)
        weights = tf.nn.softmax(weights, axis=-2)
        return tf.reduce_sum(inputs * weights, axis=-2)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-2]) + (input_shape[-1],)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_hat_ids(
    window_size: int = 20,
    num_features: int = 5,
    num_classes: int = 6,
    d_model: int = 32,
    num_heads: int = 2,
    dim_feedforward: int = 64,
    dropout: float = 0.2,
    batch_size: int | None = None,
) -> keras.Model:
    """Build the HAT-IDS (Hierarchical Attention Transformer) model.

    Args:
        window_size: Number of CAN frames in sliding window (default 20).
        num_features: Number of temporal engineered features (default 5).
        num_classes: Number of classification classes.
        d_model: Hidden dimension for nibbles and sequence representations (default 32).
        num_heads: Number of attention heads in Transformer blocks (default 2).
        dim_feedforward: Hidden dimension of FFN in Transformer blocks (default 64).
        dropout: Dropout rate (default 0.2).
        batch_size: Optional fixed batch size.
    """
    tokens_input = keras.Input(
        shape=(window_size, 19),
        batch_size=batch_size,
        dtype=tf.int32,
        name="tokens_input",
    )
    features_input = keras.Input(
        shape=(window_size, num_features),
        batch_size=batch_size,
        dtype=tf.float32,
        name="features_input",
    )
    time_embeddings = keras.Input(
        shape=(window_size, d_model),
        batch_size=batch_size,
        dtype=tf.float32,
        name="time_embeddings",
    )

    # -------------------------------------------------------------
    # 1. Level 1: Intra-Frame Spatial Transformer Submodule
    # -------------------------------------------------------------
    frame_in = keras.Input(shape=(19,), dtype=tf.int32, name="frame_input")
    nibble_emb = layers.Embedding(
        16, d_model, name="nibble_lookup_table"
    )(frame_in)  # [B, 19, d_model]
    with_cls = CLSToken(d_model=d_model, name="cls_token")(nibble_emb)  # [B, 20, d_model]
    pos = layers.Embedding(20, d_model, name="intra_pos_embed")(tf.range(20))  # [20, d_model]
    x = with_cls + pos

    key_dim = max(1, d_model // num_heads)
    attn_1 = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout, name="intra_mha"
    )(x, x)
    attn_1 = layers.Dropout(dropout, name="intra_mha_dropout")(attn_1)
    x = layers.LayerNormalization(epsilon=1e-5, name="intra_ln1")(x + attn_1)

    ffn_1 = layers.Dense(dim_feedforward, activation="relu", name="intra_ffn1")(x)
    ffn_1 = layers.Dropout(dropout, name="intra_ffn_dropout")(ffn_1)
    ffn_1 = layers.Dense(d_model, name="intra_ffn2")(ffn_1)
    x = layers.LayerNormalization(epsilon=1e-5, name="intra_ln2")(x + ffn_1)

    frame_summary = ExtractCLS(name="extract_cls")(x)  # [B, d_model]

    frame_module = keras.Model(
        inputs=frame_in, outputs=frame_summary, name="frame_transformer_submodule"
    )
    spatial_seq = layers.TimeDistributed(
        frame_module, name="time_distributed_spatial"
    )(tokens_input)  # [B, W, d_model]

    # -------------------------------------------------------------
    # 2. Multi-Scale Linear Feature Fusion (Concatenate + Dense)
    # -------------------------------------------------------------
    fused = layers.Concatenate(axis=-1, name="concat_features")(
        [spatial_seq, features_input]
    )  # [B, W, d_model + num_features]
    fused_proj = layers.Dense(
        d_model, activation="relu", name="fusion_dense"
    )(fused)  # [B, W, d_model]

    # -------------------------------------------------------------
    # 3. Time-Positional Encoding (TSE) Injection
    # -------------------------------------------------------------
    temporal_tokens = layers.Add(name="tse_add")([fused_proj, time_embeddings])
    temporal_tokens = layers.LayerNormalization(
        epsilon=1e-5, name="temporal_input_ln"
    )(temporal_tokens)

    # -------------------------------------------------------------
    # 4. Level 2: Inter-Frame Temporal Transformer Encoder
    # -------------------------------------------------------------
    temp_key_dim = max(1, d_model // num_heads)
    attn_2 = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=temp_key_dim,
        dropout=dropout,
        name="inter_mha",
    )(temporal_tokens, temporal_tokens)
    attn_2 = layers.Dropout(dropout, name="inter_mha_dropout")(attn_2)
    x2 = layers.LayerNormalization(epsilon=1e-5, name="inter_ln1")(
        temporal_tokens + attn_2
    )

    ffn_2 = layers.Dense(dim_feedforward, activation="relu", name="inter_ffn1")(x2)
    ffn_2 = layers.Dropout(dropout, name="inter_ffn_dropout")(ffn_2)
    ffn_2 = layers.Dense(d_model, name="inter_ffn2")(ffn_2)
    x2 = layers.LayerNormalization(epsilon=1e-5, name="inter_ln2")(x2 + ffn_2)

    # -------------------------------------------------------------
    # 5. Temporal Attention Pooling
    # -------------------------------------------------------------
    seq_summary = AttentionPooling1D(
        units=d_model, name="temporal_attention_pool"
    )(x2)  # [B, d_model]

    # -------------------------------------------------------------
    # 6. Classification Head
    # -------------------------------------------------------------
    outputs = layers.Dense(
        units=int(num_classes),
        activation="softmax",
        name="classification_head",
    )(seq_summary)

    return keras.Model(
        inputs=[tokens_input, features_input, time_embeddings],
        outputs=outputs,
        name="hat_ids",
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
