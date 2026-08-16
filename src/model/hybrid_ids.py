import tensorflow as tf
from tensorflow import keras
from keras import layers

from .spatial_transformer import SpatialTransformer
from .temporal_gru import TemporalGRU


class HybridIDS(keras.Model):
    """
    Hybrid Spatial Transformer + Temporal GRU IDS


    Input:

        tokens:
            (B,T,10)

        token_types:
            (B,T,10)

        positions:
            (B,T,10)

        temporal_features:
            (B,T,9)


    Flow:

        CAN frame
             |
             v
        Spatial Transformer
             |
        z_spatial

             +
        temporal features

             |
             v

            GRU

             |
             v

        classifier


    Output:

        logits:
            (B,num_classes)

    """

    def __init__(
        self,
        d_model=32,
        num_heads=2,
        ff_dim=64,
        num_layers=1,
        gru_units=64,
        num_classes=5,
        dropout=0.1,
        **kwargs,
    ):

        super().__init__(**kwargs)

        #
        # Spatial encoder
        #
        # classifier tidak digunakan
        #
        self.spatial_encoder = SpatialTransformer(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

        #
        # Temporal encoder
        #
        self.temporal_encoder = TemporalGRU(
            units=gru_units,
            dropout=dropout,
        )

        #
        # classifier
        #

        self.classifier = keras.Sequential(
            [
                layers.Dense(
                    64,
                    activation="gelu",
                ),
                layers.Dropout(dropout),
                layers.Dense(
                    num_classes,
                    activation="softmax",
                ),
            ],
            name="classifier",
        )

    def call(
        self,
        inputs,
        training=False,
    ):

        tokens = inputs["tokens"]

        token_types = inputs["token_types"]

        positions = inputs["positions"]

        temporal_features = inputs["temporal_features"]

        #
        # Shape
        #
        # B = batch
        # T = sequence length
        #

        B = tf.shape(tokens)[0]

        T = tf.shape(tokens)[1]

        #
        # Flatten sequence
        #
        # sebelum transformer:
        #
        # (B,T,10)
        #
        # menjadi
        #
        # (B*T,10)
        #

        frame_tokens = tf.reshape(tokens, (-1, 10))

        frame_types = tf.reshape(token_types, (-1, 10))

        frame_positions = tf.reshape(positions, (-1, 10))

        #
        # Spatial extraction
        #
        # output:
        #
        # (B*T,d_model)
        #

        z = self.spatial_encoder(
            {
                "tokens": frame_tokens,
                "token_types": frame_types,
                "positions": frame_positions,
            },
            training=training,
            return_embedding=True,
        )

        #
        # kembali ke sequence
        #
        # (B*T,d)
        #
        # menjadi
        #
        # (B,T,d)
        #

        z = tf.reshape(
            z,
            (
                B,
                T,
                -1,
            ),
        )

        #
        # gabungkan spatial + temporal feature
        #
        # (B,T,d_model+9)
        #

        x = tf.concat(
            [
                z,
                temporal_features,
            ],
            axis=-1,
        )

        #
        # GRU
        #
        # output:
        #
        # (B,gru_units)
        #

        h = self.temporal_encoder(
            x,
            training=training,
        )

        #
        # classifier
        #

        output = self.classifier(
            h,
            training=training,
        )

        return output
