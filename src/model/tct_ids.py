"""TensorFlow/Keras implementation of TCT-IDS.

Reference:
Gong et al. (2026), "TCT-IDS: A Temporal-Convolutional and Transformer-Based
Intrusion Detection System for In-Vehicle CAN Networks", IEEE ICCET 2026.
"""

import os
import types

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="tct_ids")
class GlobalSumPooling1D(layers.Layer):
    """Global sum-pooling across temporal sequence dimension."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


@keras.utils.register_keras_serializable(package="tct_ids")
class CausalConv1D(layers.Layer):
    """Dilated Causal 1D Convolution with Weight Normalization (Fig. 3).

    Paper Fig. 3 explicitly specifies 'Weighted Norm' after Dilated Causal Convolution.
    Weight normalization decomposes weights into magnitude g and direction v/||v||:
        w = g * (v / ||v||)
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 2,
        dilation_rate: int = 1,
        use_weight_norm: bool = True,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.dilation_rate = int(dilation_rate)
        self.use_weight_norm = bool(use_weight_norm)
        self.use_bias = bool(use_bias)

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, in_channels, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None

        if self.use_weight_norm:
            self.g = self.add_weight(
                name="g",
                shape=(self.filters,),
                initializer="ones",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs):
        pad_len = self.dilation_rate * (self.kernel_size - 1)
        if pad_len > 0:
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])

        weight = self.kernel
        if self.use_weight_norm:
            norm = tf.sqrt(
                tf.reduce_sum(tf.square(weight), axis=[0, 1], keepdims=True)
                + 1e-8
            )
            weight = weight / norm * tf.reshape(self.g, (1, 1, self.filters))

        out = tf.nn.conv1d(
            inputs,
            weight,
            stride=1,
            padding="VALID",
            dilations=self.dilation_rate,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "use_weight_norm": self.use_weight_norm,
                "use_bias": self.use_bias,
            }
        )
        return config


def enable_rdrop_training(
    model: keras.Model, lambda_rdrop: float = 0.1
) -> keras.Model:
    """Attach R-Drop regularized training step to a compiled Keras model.

    R-Drop passes each input twice with independent Dropout masks and penalizes
    the symmetric Kullback-Leibler (KL) divergence between the two output distributions:
        L_total = 0.5 * (L_ce1 + L_ce2) + lambda_rdrop * L_kl
    """

    def rdrop_train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # 1. Two stochastic forward passes under active dropout
            p1 = self(x, training=True)
            p2 = self(x, training=True)

            # 2. Average Cross-Entropy loss
            loss_ce1 = self.compiled_loss(
                y, p1, regularization_losses=self.losses
            )
            loss_ce2 = self.compiled_loss(y, p2)
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            # 3. Symmetric KL divergence
            eps = 1e-8
            p1_c = tf.clip_by_value(p1, eps, 1.0)
            p2_c = tf.clip_by_value(p2, eps, 1.0)
            kl_12 = tf.reduce_sum(
                p1_c * (tf.math.log(p1_c) - tf.math.log(p2_c)), axis=-1
            )
            kl_21 = tf.reduce_sum(
                p2_c * (tf.math.log(p2_c) - tf.math.log(p1_c)), axis=-1
            )
            loss_kl = 0.5 * tf.reduce_mean(kl_12 + kl_21)

            # 4. Total regularized loss
            total_loss = loss_ce + float(lambda_rdrop) * loss_kl

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        # Update metrics tracking using primary prediction
        self.compiled_metrics.update_state(y, p1)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = total_loss
        results["loss_kl"] = loss_kl
        return results

    model.train_step = types.MethodType(rdrop_train_step, model)
    return model


def build_tct_ids(
    window_size: int = 15,
    d_model: int = 10,
    num_heads: int = 1,
    num_layers: int = 3,
    dim_feedforward: int = 40,
    mlp_hidden_dim: int = 40,
    tcn_filters: int = 96,
    tcn_kernel_size: int = 2,
    tcn_dilations: list[int] | tuple[int, ...] = (1, 2, 4),
    use_weight_norm: bool = True,
    dropout: float = 0.1,
    num_classes: int = 6,
    fusion: str = "concat",
    pooling: str = "last",
    batch_size: int | None = None,
) -> keras.Model:
    """Build the TCT-IDS model reproducing Gong et al. (2026).

    Architecture:
    1. Packet-Level Spatial Extractor g:
       Dense(12 -> mlp_hidden_dim) -> ReLU -> Dropout -> Dense(d_model) -> mese
    2. Sequence-Level Temporal Extractor f (TCN with Weighted Norm):
       Dilated causal Conv1D residual blocks (dilations 1, 2, 4) with WeightNorm.
       Last sequence step extracted and projected to d_model: X^t = mest_last
    3. Multi-Scale Feature Fusion:
       Concatenation (Eq. 8: X^et = mese || X^t) projected to d_model, or element-wise addition.
    4. Time-Positional Encoding injection:
       Add sinusoidal timestamp encoding (TSE): X = X^et + TSE
    5. Transformer Encoder stack:
       num_layers Post-LN Transformer Encoder layers (MHA + FFN)
    6. Classification Head:
       Pooling (Last step / Mean / Sum) -> Dense(num_classes, softmax)
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

    # 1. Packet-Level Spatial Extractor (MLP)
    mlp = layers.Dense(
        mlp_hidden_dim, activation="relu", name="mlp_dense_1"
    )(message_input)
    mlp = layers.Dropout(dropout, name="mlp_dropout_1")(mlp)
    mese = layers.Dense(d_model, activation="relu", name="mlp_dense_2")(mlp)

    # 2. Sequence-Level Temporal Extractor (TCN with Weight Normalization)
    tcn_in = message_input
    for idx, d in enumerate(tcn_dilations, 1):
        conv1 = CausalConv1D(
            filters=tcn_filters,
            kernel_size=tcn_kernel_size,
            dilation_rate=d,
            use_weight_norm=use_weight_norm,
            name=f"tcn_conv1_{idx}",
        )(tcn_in)
        act1 = layers.Activation("relu", name=f"tcn_act1_{idx}")(conv1)
        drop1 = layers.Dropout(dropout, name=f"tcn_drop1_{idx}")(act1)

        conv2 = CausalConv1D(
            filters=tcn_filters,
            kernel_size=tcn_kernel_size,
            dilation_rate=d,
            use_weight_norm=use_weight_norm,
            name=f"tcn_conv2_{idx}",
        )(drop1)
        act2 = layers.Activation("relu", name=f"tcn_act2_{idx}")(conv2)
        drop2 = layers.Dropout(dropout, name=f"tcn_drop2_{idx}")(act2)

        if tcn_in.shape[-1] != tcn_filters:
            shortcut = layers.Conv1D(
                filters=tcn_filters,
                kernel_size=1,
                padding="same",
                name=f"tcn_shortcut_{idx}",
            )(tcn_in)
        else:
            shortcut = tcn_in

        tcn_in = layers.Add(name=f"tcn_add_{idx}")([drop2, shortcut])

    # Extract representation from the last sequence step (Eq. 6)
    tcn_last = layers.Lambda(
        lambda t: t[:, -1, :], name="tcn_last_step"
    )(tcn_in)

    # Project TCN representation to d_model
    tcn_proj = layers.Dense(d_model, name="tcn_projection")(tcn_last)

    # 3. Multi-Scale Feature Fusion
    if fusion == "concat":
        # Eq. 8: X^{et} = {mes_i^e}_{i=1}^n || X^t
        tcn_repeated = layers.RepeatVector(
            window_size, name="tcn_repeat"
        )(tcn_proj)
        fused = layers.Concatenate(axis=-1, name="fuse_concat")([
            mese,
            tcn_repeated,
        ])
        fused = layers.Dense(d_model, name="fuse_projection")(fused)
    else:
        # Fig. 1 addition: X^{et} = g(X^p) + f(X^p)
        tcn_expanded = layers.Reshape(
            (1, d_model), name="tcn_expand"
        )(tcn_proj)
        fused = layers.Add(name="fuse_mlp_tcn_add")([mese, tcn_expanded])

    # 4. Time-Positional Encoding injection (Eq. 9, 10)
    x = layers.Add(name="tse_add")([fused, time_input])

    # 5. Transformer Encoder stack
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
        res_attn = layers.Add(name=f"transformer_layer_{layer_idx}_attn_res")([
            x,
            attn_out,
        ])
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
        res_ffn = layers.Add(name=f"transformer_layer_{layer_idx}_ffn_res")([
            x,
            ffn,
        ])
        x = layers.LayerNormalization(
            epsilon=1e-5, name=f"transformer_layer_{layer_idx}_ffn_norm"
        )(res_ffn)

    # 6. Global Pooling
    if pooling == "mean":
        pooled = layers.GlobalAveragePooling1D(name="mean_pooling")(x)
    elif pooling == "sum":
        pooled = GlobalSumPooling1D(name="sum_pooling")(x)
    else:
        # Default: Last-token sequence representation (c = FC(X^{en}))
        pooled = layers.Lambda(
            lambda t: t[:, -1, :], name="last_step_pooling"
        )(x)

    # 7. Classification Output
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
        name="tct_ids",
    )


def parameter_counts(model: keras.Model) -> dict[str, int]:
    """Breakdown of trainable and total parameters by component."""

    def count(weights):
        return int(sum(int(p.numpy().size) for p in weights))

    mlp_layers = [
        layer for layer in model.layers if layer.name.startswith("mlp_dense")
    ]
    tcn_layers = [
        layer
        for layer in model.layers
        if layer.name.startswith("tcn_")
    ]
    transformer_layers = [
        layer
        for layer in model.layers
        if layer.name.startswith("transformer_layer")
    ]
    classifier_layers = [
        layer
        for layer in model.layers
        if layer.name.startswith("classification_")
    ]

    mlp_params = count([w for layer in mlp_layers for w in layer.weights])
    tcn_params = count([w for layer in tcn_layers for w in layer.weights])
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
        "mlp_parameters": mlp_params,
        "tcn_parameters": tcn_params,
        "transformer_parameters": transformer_params,
        "classifier_parameters": classifier_params,
    }
