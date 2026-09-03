import tensorflow as tf
from tensorflow import keras
from keras import layers

from .embeddings import NumericFeatureProjection
from .transformer import TransformerEncoder


class SpatialTransformer(keras.Model):
    def __init__(
        self,
        d_model=32,
        num_heads=2,
        ff_dim=64,
        num_layers=1,
        num_classes=5,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        #
        # Numeric Encoding + Feature Projection
        #
        self.embedding = NumericFeatureProjection(
            d_model=d_model,
            max_position=10,
            dropout=dropout,
        )

        #
        # Spatial Attention
        #
        self.encoder_layers = [
            TransformerEncoder(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]

        #
        # classifier hanya untuk pretraining
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
            ]
        )

    def call(
        self,
        inputs,
        training=False,
        return_embedding=False,
    ):

        #
        # Numeric Encoding -> Feature Projection
        #
        x = self.embedding(
            inputs,
            training=training,
        )

        #
        # Spatial Attention
        #
        for encoder in self.encoder_layers:
            x = encoder(
                x,
                training=training,
            )

        #
        # z_spatial (masked mean pooling, abaikan slot PAD)
        #
        numeric_values = inputs["numeric_values"]  # (..., 10)

        valid_mask = tf.cast(
            tf.greater_equal(numeric_values, 0.0),
            dtype=x.dtype,
        )  # (..., 10)

        valid_mask = tf.expand_dims(valid_mask, axis=-1)  # (..., 10, 1)

        masked_sum = tf.reduce_sum(x * valid_mask, axis=-2)  # (..., d_model)
        valid_count = tf.reduce_sum(valid_mask, axis=-2)  # (..., 1)
        valid_count = tf.maximum(valid_count, 1.0)  # jaga-jaga div by zero

        z_spatial = masked_sum / valid_count

        if return_embedding:
            return z_spatial

        #
        # classification head
        #
        output = self.classifier(
            z_spatial,
            training=training,
        )

        return output
