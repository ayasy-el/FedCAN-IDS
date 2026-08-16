import tensorflow as tf

from tensorflow import keras

from data.dataset import CANDataset
from model.spatial_transformer import SpatialTransformer



# ==========================
# Configuration
# ==========================

TRAIN_PATH = (
    "data/processed/car_hacking/train.parquet"
)

VAL_PATH = (
    "data/processed/car_hacking/val.parquet"
)


BATCH_SIZE = 256

EPOCHS = 50



# ==========================
# Dataset
# ==========================

train_dataset = CANDataset(
    TRAIN_PATH,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


val_dataset = CANDataset(
    VAL_PATH,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


train_tf = train_dataset.to_tf_dataset()

val_tf = val_dataset.to_tf_dataset()



# ==========================
# Model
# ==========================

model = SpatialTransformer(
    d_model=4,
    num_heads=2,
    ff_dim=8,
    num_layers=1,
    num_classes=5,
    dropout=0.1,
)


# build model

dummy = {
    "tokens": tf.zeros(
        (1,10),
        dtype=tf.int32,
    ),

    "token_types": tf.zeros(
        (1,10),
        dtype=tf.int32,
    ),

    "positions": tf.zeros(
        (1,10),
        dtype=tf.int32,
    ),
}


model(dummy)

model.summary()



# ==========================
# Compile
# ==========================


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ],
)



# ==========================
# Callback
# ==========================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=
        "checkpoints/spatial_best.keras",

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",
    ),


    keras.callbacks.EarlyStopping(
        monitor="val_loss",

        patience=5,

        restore_best_weights=True,
    ),


    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",

        factor=0.5,

        patience=3,
    ),
]



# ==========================
# Training
# ==========================

history = model.fit(
    train_tf,

    validation_data=val_tf,

    epochs=EPOCHS,

    callbacks=callbacks,
)