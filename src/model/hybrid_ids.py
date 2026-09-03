import tensorflow as tf
from tensorflow import keras
from keras import layers

from .spatial_transformer import SpatialTransformer
from .temporal_gru import TemporalGRU


class HybridIDS(keras.Model):
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

        numeric_values = inputs["numeric_values"]

        positions = inputs["positions"]

        temporal_features = inputs["temporal_features"]

        #
        # Shape
        #
        # B = batch
        # T = sequence length
        #

        B = tf.shape(numeric_values)[0]

        T = tf.shape(numeric_values)[1]

        #
        # Flatten sequence
        #

        frame_numeric_values = tf.reshape(numeric_values, (-1, 10))

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
                "numeric_values": frame_numeric_values,
                "positions": frame_positions,
            },
            training=training,
            return_embedding=True,
        )

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
