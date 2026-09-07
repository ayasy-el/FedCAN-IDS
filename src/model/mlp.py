"""Dense MLP baseline for frame-level CAN intrusion classification."""

from tensorflow import keras
from keras import layers


def build_mlp(input_dim: int, n_classes: int) -> keras.Model:
    data_input = keras.Input(shape=(input_dim,), name="data_input")
    hidden_1 = layers.Dense(16, activation="relu", name="hidden_1")(data_input)
    hidden_2 = layers.Dense(8, activation="relu", name="hidden_2")(hidden_1)
    hidden_3 = layers.Dense(8, activation="relu", name="hidden_3")(hidden_2)
    data_output = layers.Dense(n_classes, activation="softmax", name="data_output")(hidden_3)
    return keras.Model(data_input, data_output, name="mlp")
