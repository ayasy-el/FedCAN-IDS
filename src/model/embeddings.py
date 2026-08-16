import tensorflow as tf
from tensorflow import keras
from keras import layers


class CANEmbedding(layers.Layer):
    """
    Input:
        tokens       : (B, 10)
        token_types  : (B, 10)
        positions    : (B, 10)

    Output:
        embeddings   : (B, 10, d_model)
    """

    def __init__(
        self,
        d_model=4,
        id_vocab_size=2048,
        dlc_vocab_size=9,
        byte_vocab_size=257,
        type_vocab_size=4,
        max_position=10,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_model = d_model

        #
        # Token embeddings
        #
        self.id_embedding = layers.Embedding(
            input_dim=id_vocab_size,
            output_dim=d_model,
            name="id_embedding",
        )

        self.dlc_embedding = layers.Embedding(
            input_dim=dlc_vocab_size,
            output_dim=d_model,
            name="dlc_embedding",
        )

        self.byte_embedding = layers.Embedding(
            input_dim=byte_vocab_size,
            output_dim=d_model,
            name="byte_embedding",
        )

        #
        # Type embedding
        #
        self.type_embedding = layers.Embedding(
            input_dim=type_vocab_size,
            output_dim=d_model,
            name="type_embedding",
        )

        #
        # Position embedding
        #
        self.position_embedding = layers.Embedding(
            input_dim=max_position,
            output_dim=d_model,
            name="position_embedding",
        )

        self.dropout = layers.Dropout(
            dropout
        )

    def call(
        self,
        inputs,
        training=None,
    ):
        tokens = inputs["tokens"]
        token_types = inputs["token_types"]
        positions = inputs["positions"]

        #
        # Split token
        #
        id_token = tokens[:, 0]
        dlc_token = tokens[:, 1]
        byte_tokens = tokens[:, 2:]

        #
        # Token embeddings
        #
        id_emb = self.id_embedding(
            id_token
        )
        dlc_emb = self.dlc_embedding(
            dlc_token
        )
        byte_emb = self.byte_embedding(
            byte_tokens
        )

        #
        # Rebuild sequence
        #
        x = tf.concat(
            [
                tf.expand_dims(
                    id_emb,
                    axis=1,
                ),
                tf.expand_dims(
                    dlc_emb,
                    axis=1,
                ),
                byte_emb,
            ],
            axis=1,
        )

        #
        # Add type embedding
        #
        type_emb = self.type_embedding(
            token_types
        )

        #
        # Add positional embedding
        #
        pos_emb = self.position_embedding(
            positions
        )

        x = (
            x
            + type_emb
            + pos_emb
        )

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