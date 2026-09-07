"""Dense MLP classifier for flattened causal CAN frame windows."""

from keras import layers
from tensorflow import keras


def build_mlp_window(input_dim: int, n_classes: int, hidden_dims: list[int], dropout: float = 0.2) -> keras.Model:
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer")

    data_input = keras.Input(shape=(input_dim,), name="window_input")
    hidden = data_input
    for index, units in enumerate(hidden_dims):
        hidden = layers.Dense(units, activation="relu", name=f"hidden_{index + 1}")(hidden)
        if index == 0:
            hidden = layers.LayerNormalization(epsilon=1e-6, name="hidden_1_norm")(hidden)
        if dropout:
            hidden = layers.Dropout(dropout, name=f"dropout_{index + 1}")(hidden)
    output = layers.Dense(n_classes, activation="softmax", name="data_output")(hidden)
    return keras.Model(data_input, output, name="mlp_window")
