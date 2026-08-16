from tensorflow import keras
from keras import layers


class TransformerEncoder(
    layers.Layer
):
    """
    Input:
        (B, T, d_model)

    Output:
        (B, T, d_model)
    """

    def __init__(
        self,
        d_model=4,
        num_heads=2,
        ff_dim=8,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attention = (
            layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model
                // num_heads,
                dropout=dropout,
                name="mha",
            )
        )

        self.dropout1 = (
            layers.Dropout(
                dropout
            )
        )

        self.norm1 = (
            layers.LayerNormalization(
                epsilon=1e-6
            )
        )

        self.ffn = keras.Sequential(
            [
                layers.Dense(
                    ff_dim,
                    activation="gelu",
                ),
                layers.Dense(
                    d_model,
                ),
            ],
            name="ffn",
        )

        self.dropout2 = (
            layers.Dropout(
                dropout
            )
        )

        self.norm2 = (
            layers.LayerNormalization(
                epsilon=1e-6
            )
        )

    def call(
        self,
        inputs,
        training=None,
    ):
        #
        # Self-attention
        #
        attn = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
        )

        attn = self.dropout1(
            attn,
            training=training,
        )

        #
        # Add & Norm
        #
        x = self.norm1(
            inputs + attn
        )

        #
        # Feed-forward
        #
        ffn = self.ffn(
            x,
            training=training,
        )

        ffn = self.dropout2(
            ffn,
            training=training,
        )

        #
        # Add & Norm
        #
        x = self.norm2(
            x + ffn
        )

        return x

    def get_config(self):
        config = super().get_config()
        return config