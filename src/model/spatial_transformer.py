import tensorflow as tf
from tensorflow import keras
from keras import layers

from .embeddings import CANEmbedding
from .transformer import TransformerEncoder


class SpatialTransformer(keras.Model):
    """
    Lightweight Transformer for CAN spatial feature extraction.

    Input:
        tokens
            (B,10)

        token_types
            (B,10)

        positions
            (B,10)


    Output:
        logits
            (B,num_classes)

        z_spatial
            (B,d_model)
    """

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


        self.embedding = CANEmbedding(
            d_model=d_model,

            # Classical CAN
            id_vocab_size=2048,
            dlc_vocab_size=9,
            byte_vocab_size=257,

            type_vocab_size=4,
            max_position=10,

            dropout=dropout,
        )


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
        # spatial representation
        #
        # CATATAN (fix bug PAD masking): GlobalAveragePooling1D polos
        # merata-ratakan SEMUA 10 token termasuk token PAD (byte kosong
        # untuk DLC < 8), sehingga mengencerkan representasi payload
        # pendek. Pooling PAD-aware diimplementasikan manual di call()
        # di bawah -- layer ini tidak lagi dipakai, disimpan hanya untuk
        # kompatibilitas checkpoint lama kalau ada.
        self.pooling = layers.GlobalAveragePooling1D()


        #
        # classifier hanya untuk pretraining
        #
        self.classifier = keras.Sequential(
            [
                layers.Dense(
                    64,
                    activation="gelu",
                ),

                layers.Dropout(
                    dropout
                ),

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
        # token embedding
        #
        x = self.embedding(
            inputs,
            training=training,
        )


        #
        # Transformer encoder
        #
        for encoder in self.encoder_layers:
            x = encoder(
                x,
                training=training,
            )


        #
        # contextual spatial embedding (masked pooling, abaikan token PAD)
        #
        # token_types: 0=ID, 1=DLC, 2=byte payload (indeks 2..9)
        # tokens byte bernilai 256 -> PAD (lihat byte_vocab_size=257 di
        # embeddings.py dan payload_exprs di temporal_dataset.py/dataset.py)
        #
        # ID/DLC token TIDAK PERNAH di-mask (selalu valid), hanya token
        # byte payload dengan nilai PAD yang dikecualikan dari rata-rata,
        # supaya frame dengan DLC<8 tidak "diencerkan" representasinya
        # oleh byte kosong.
        #
        tokens = inputs["tokens"]
        token_types = inputs["token_types"]

        is_pad_byte = tf.logical_and(
            tf.equal(token_types, 2),
            tf.equal(tokens, 256),
        )

        valid_mask = tf.cast(
            tf.logical_not(is_pad_byte),
            dtype=x.dtype,
        )  # (..., 10)

        valid_mask = tf.expand_dims(valid_mask, axis=-1)  # (..., 10, 1)

        masked_sum = tf.reduce_sum(x * valid_mask, axis=-2)  # (..., d_model)
        valid_count = tf.reduce_sum(valid_mask, axis=-2)     # (..., 1)
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