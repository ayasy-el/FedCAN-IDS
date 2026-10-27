"""TensorFlow/Keras implementation of Hierarchical Spatio-Temporal Transformer-GRU with Nibble Lookup Table.

Architecture Pipeline:
1. Frame Nibbles Input: [B, W, 19] (int32) - 19 nibbles per frame in range [0..15]:
   - nibbles 0..2: 3 CAN Arbitration ID nibbles (12 bits)
   - nibbles 3..18: 16 payload nibbles (upper & lower 4 bits for Data_0..Data_7)
2. Temporal Features Input: [B, W, N_feat] (float32) - 5 engineered CAN temporal features
3. Intra-Frame Spatial Transformer Submodule (per frame):
   - Shared Nibble Lookup Table: Embedding(16, d_model) -> [19, d_model]
   - Trainable [CLS] token prepended at index 0 -> [20, d_model]
   - Positional Embedding for tokens 0..19 -> [20, d_model]
   - MultiHeadAttention(num_heads, key_dim) + LayerNorm + MLP + LayerNorm
   - Frame Summary: Extracted [CLS] token at index 0 -> [d_model]
4. Gated Multimodal Feature Fusion:
   - Dynamic gating between spatial frame summary and 5 temporal features:
     z = sigmoid(W_s * S + W_f * F + b)
     H = z * tanh(W_s * S) + (1 - z) * tanh(W_f * F) -> [B, W, d_model]
5. Recurrent Temporal Modeling:
   - GRU(gru_hidden_size, return_sequences=True) -> [B, W, gru_hidden_size]
6. Inter-Frame Temporal Attention:
   - MultiHeadAttention across full sliding window timesteps W -> [B, W, gru_hidden_size]
   - AttentionPooling1D across time dimension -> [B, gru_hidden_size]
7. Classification Head:
   - Dense(num_classes, activation='softmax') -> [B, num_classes]
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="hierarchical_transformer_gru")
class CLSToken(layers.Layer):
    """Prepends a trainable [CLS] token to the sequence."""

    def __init__(self, d_model: int = 16, **kwargs):
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


@keras.utils.register_keras_serializable(package="hierarchical_transformer_gru")
class ExtractCLS(layers.Layer):
    """Extracts the [CLS] token representation at index 0."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


@keras.utils.register_keras_serializable(package="hierarchical_transformer_gru")
class NibbleFrameTokenizer(layers.Layer):
    """Embeds 19 CAN frame nibbles using a 16-entry Nibble Lookup Table and prepends [CLS]."""

    def __init__(self, d_model: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.nibble_lookup = layers.Embedding(
            16, self.d_model, name="nibble_lookup_table"
        )
        self.pos_embed = layers.Embedding(20, self.d_model, name="pos_embed")
        self.cls_layer = CLSToken(d_model=self.d_model, name="cls_token")

    def call(self, frame_tokens):
        # frame_tokens: [B, 19] (values 0..15)
        # 3 CAN ID nibbles + 16 payload nibbles
        nibble_emb = self.nibble_lookup(frame_tokens)  # [B, 19, d_model]
        with_cls = self.cls_layer(nibble_emb)          # [B, 20, d_model]
        pos_idx = tf.range(20)
        pos = self.pos_embed(pos_idx)                  # [20, d_model]
        return with_cls + pos

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 20, self.d_model)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


@keras.utils.register_keras_serializable(package="hierarchical_transformer_gru")
class AttentionPooling1D(layers.Layer):
    """Learned context-aware attention pooling across sequence dimension."""

    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.score_dense = layers.Dense(self.units, activation="tanh", name="score_tanh")
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


@keras.utils.register_keras_serializable(package="hierarchical_transformer_gru")
class GatedFusion(layers.Layer):
    """Gated Multimodal Fusion between frame representation and engineered features."""

    def __init__(self, output_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = int(output_dim)
        self.proj_spatial = layers.Dense(self.output_dim, activation="tanh", name="proj_spatial")
        self.proj_features = layers.Dense(self.output_dim, activation="tanh", name="proj_features")
        self.gate_dense = layers.Dense(self.output_dim, activation="sigmoid", name="gate_dense")

    def call(self, inputs):
        spatial, features = inputs
        s = self.proj_spatial(spatial)
        f = self.proj_features(features)
        z = self.gate_dense(tf.concat([spatial, features], axis=-1))
        return z * s + (1.0 - z) * f

    def compute_output_shape(self, input_shapes):
        return tuple(input_shapes[0][:-1]) + (self.output_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


def build_hierarchical_transformer_gru(
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
    """Build Hierarchical Spatio-Temporal Transformer-GRU model with Nibble Lookup Table.

    Args:
        window_size: Number of CAN frames in sliding window (default 20).
        num_features: Number of temporal engineered features (default 5).
        num_classes: Number of classification classes (default 6).
        d_model: Hidden dimension for nibble tokens and frame representation (default 16).
        num_heads: Number of attention heads (default 2).
        dim_feedforward: Hidden dimension of FFN in Transformer block (default 32).
        dropout: Dropout rate (default 0.2).
        gru_hidden_size: Hidden units in temporal GRU layer (default 32).
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

    # -------------------------------------------------------------
    # 1. Intra-Frame Transformer Submodule (per frame)
    # -------------------------------------------------------------
    frame_in = keras.Input(shape=(19,), dtype=tf.int32, name="frame_input")

    x = NibbleFrameTokenizer(
        d_model=d_model,
        name="nibble_tokenizer",
    )(frame_in)  # [B, 20, d_model]

    key_dim = max(1, d_model // num_heads)
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name="spatial_mha",
    )(x, x)
    attn_out = layers.Dropout(dropout, name="spatial_mha_dropout")(attn_out)
    x = layers.LayerNormalization(epsilon=1e-5, name="spatial_ln1")(x + attn_out)

    ffn = layers.Dense(dim_feedforward, activation="relu", name="spatial_ffn1")(x)
    ffn = layers.Dropout(dropout, name="spatial_ffn_dropout")(ffn)
    ffn = layers.Dense(d_model, name="spatial_ffn2")(ffn)
    x = layers.LayerNormalization(epsilon=1e-5, name="spatial_ln2")(x + ffn)

    # Extract [CLS] token (token 0) as full-frame semantic summary
    frame_summary = ExtractCLS(name="extract_cls")(x)  # [B, d_model]

    frame_module = keras.Model(
        inputs=frame_in,
        outputs=frame_summary,
        name="frame_transformer_submodule",
    )

    # Apply across all window timesteps -> [B, W, d_model]
    frame_seq = layers.TimeDistributed(
        frame_module, name="time_distributed_frame"
    )(tokens_input)

    # -------------------------------------------------------------
    # 2. Gated Multimodal Feature Fusion
    # -------------------------------------------------------------
    fused_seq = GatedFusion(
        output_dim=d_model, name="gated_fusion"
    )([frame_seq, features_input])

    # -------------------------------------------------------------
    # 3. Recurrent Temporal Modeling (GRU with return_sequences=True)
    # -------------------------------------------------------------
    rnn_seq = layers.GRU(
        units=int(gru_hidden_size),
        return_sequences=True,
        name="gru",
    )(fused_seq)  # [B, W, gru_hidden_size]

    # -------------------------------------------------------------
    # 4. Inter-Frame Temporal Attention (Self-Attention across time)
    # -------------------------------------------------------------
    temp_key_dim = max(1, gru_hidden_size // num_heads)
    temp_attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=temp_key_dim,
        dropout=dropout,
        name="temporal_mha",
    )(rnn_seq, rnn_seq)
    temp_attn = layers.Dropout(dropout, name="temporal_mha_dropout")(temp_attn)
    temp_repr = layers.LayerNormalization(epsilon=1e-5, name="temporal_ln")(
        rnn_seq + temp_attn
    )

    # Aggregate sequence across time W -> [B, gru_hidden_size]
    temporal_summary = AttentionPooling1D(
        units=gru_hidden_size, name="temporal_attention_pool"
    )(temp_repr)

    # -------------------------------------------------------------
    # 5. Classification Head
    # -------------------------------------------------------------
    outputs = layers.Dense(
        units=int(num_classes),
        activation="softmax",
        name="classification_head",
    )(temporal_summary)

    return keras.Model(
        inputs=[tokens_input, features_input],
        outputs=outputs,
        name="hierarchical_transformer_gru",
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
