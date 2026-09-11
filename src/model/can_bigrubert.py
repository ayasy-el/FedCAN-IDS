"""TensorFlow implementation of the paper's frozen-BERT BiGRU model."""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel


def build_can_bigrubert(window_size, max_length, bert_checkpoint,
                        bigru_hidden_size=512, dropout=0.1, num_classes=10):
    # The checkpoint currently resolves to Hugging Face PyTorch weights;
    # TensorFlow imports those weights into the frozen TF model.
    bert = TFBertModel.from_pretrained(bert_checkpoint, from_pt=True)
    bert.trainable = False
    hidden_size = int(bert.config.hidden_size)

    input_ids = keras.Input(
        shape=(window_size, max_length), dtype=tf.int32, name="input_ids"
    )
    attention_mask = keras.Input(
        shape=(window_size, max_length), dtype=tf.int32, name="attention_mask"
    )
    flat_ids = layers.Lambda(
        lambda tensor: tf.reshape(tensor, (-1, max_length)),
        name="flatten_frame_tokens",
    )(input_ids)
    flat_mask = layers.Lambda(
        lambda tensor: tf.reshape(tensor, (-1, max_length)),
        name="flatten_frame_masks",
    )(attention_mask)
    bert_output = bert(
        input_ids=flat_ids,
        attention_mask=flat_mask,
        training=False,
    ).last_hidden_state
    cls_embeddings = layers.Lambda(
        lambda tensor: tensor[:, 0, :], name="cls_embedding"
    )(bert_output)
    frame_sequence = layers.Lambda(
        lambda tensor: tf.reshape(tensor, (-1, window_size, hidden_size)),
        name="frame_sequence",
    )(cls_embeddings)

    temporal = layers.Bidirectional(
        layers.GRU(
            bigru_hidden_size,
            return_sequences=True,
        ),
        name="bigru_layer_1",
    )(frame_sequence)
    temporal = layers.Dropout(dropout, name="bigru_layer_1_dropout")(temporal)
    temporal = layers.Bidirectional(
        layers.GRU(bigru_hidden_size, return_sequences=False),
        name="bigru_layer_2",
    )(temporal)
    temporal = layers.Dropout(dropout, name="bigru_output_dropout")(temporal)
    hidden = layers.Dense(512, name="classifier_dense_1")(temporal)
    hidden = layers.ReLU(name="classifier_relu")(hidden)
    hidden = layers.Dense(512, name="classifier_dense_2")(hidden)
    output = layers.Dense(num_classes, activation="softmax", name="classifier_output")(hidden)
    return keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=output,
        name="can_bigrubert",
    )


def parameter_counts(model):
    def count(weights):
        return int(sum(int(parameter.numpy().size) for parameter in weights))

    bert_layers = [
        layer for layer in model.layers
        if layer.name.startswith("tf_bert_model") or layer.name.startswith("bert")
    ]
    bigru_layers = [
        layer for layer in model.layers if layer.name.startswith("bigru_layer")
    ]
    classifier_layers = [
        layer for layer in model.layers
        if layer.name.startswith("classifier_")
    ]
    bert_params = count(
        [weight for layer in bert_layers for weight in layer.weights]
    )
    bigru_params = count(
        [weight for layer in bigru_layers for weight in layer.weights]
    )
    classifier_params = count(
        [weight for layer in classifier_layers for weight in layer.weights]
    )
    return {
        "total_parameters": int(model.count_params()),
        "trainable_parameters": count(model.trainable_weights),
        "non_trainable_parameters": count(model.non_trainable_weights),
        "bert_parameters": bert_params,
        "bigru_parameters": bigru_params,
        "classifier_parameters": classifier_params,
    }
