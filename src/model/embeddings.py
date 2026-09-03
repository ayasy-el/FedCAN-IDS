import tensorflow as tf
from tensorflow import keras
from keras import layers


class NumericFeatureProjection(layers.Layer):
    def __init__(
        self,
        d_model=4,
        max_position=10,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model

        #
        # Feature Projection
        #
        self.feature_projection = layers.Dense(
            d_model,
            name="feature_projection",
        )

        #
        # Position embedding
        #
        self.position_embedding = layers.Embedding(
            input_dim=max_position,
            output_dim=d_model,
            name="position_embedding",
        )

        self.dropout = layers.Dropout(dropout)

    def call(
        self,
        inputs,
        training=None,
    ):
        numeric_values = inputs["numeric_values"]  # (B, 10)
        positions = inputs["positions"]  # (B, 10)

        #
        # Numeric Encoding -> Feature Projection
        #
        x = self.feature_projection(
            numeric_values[..., tf.newaxis]  # (B, 10, 1) -> (B, 10, d_model)
        )

        #
        # Tambah positional embedding
        #
        pos_emb = self.position_embedding(positions)

        x = x + pos_emb

        x = self.dropout(
            x,
            training=training,
        )

        return x

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "d_model": self.d_model,
            }
        )

        return config
