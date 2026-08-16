import tensorflow as tf
from tensorflow import keras
from keras import layers


class TemporalGRU(layers.Layer):

    def __init__(
        self,
        units=64,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.gru = layers.GRU(
            units,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=0.0,
            name="gru",
        )

    def call(
        self,
        x,
        training=None,
    ):
        return self.gru(
            x,
            training=training,
        )