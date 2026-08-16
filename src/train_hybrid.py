import tensorflow as tf
from tensorflow import keras

from data.temporal_dataset import TemporalCANDataset
from model.hybrid_ids import HybridIDS
from utils.class_weights import compute_class_weight_dict
from utils.metrics import MacroF1Score


# ==========================================================
# Configuration
# ==========================================================

TRAIN_PATH = "data/processed/car_hacking/featured/train.parquet"

VAL_PATH = "data/processed/car_hacking/featured/val.parquet"

SEQ_LEN = 32
BATCH_SIZE = 256
EPOCHS = 50

PRETRAINED_SPATIAL = "checkpoints/spatial_best.keras"

# Statistik normalisasi (mean/std) fitur temporal, dihitung SEKALI dari
# train, lalu dipakai ulang untuk val & test (lihat eval_hybrid.py) --
# mencegah data leakage. Lihat docstring TemporalCANDataset untuk alasan
# kenapa normalisasi ini perlu (fitur temporal skalanya sangat timpang,
# berisiko membuat gate GRU jenuh / gradient hilang).
NORMALIZE_STATS_PATH = "checkpoints/temporal_norm_stats.json"


# ==========================================================
# Dataset
# ==========================================================

train_dataset = TemporalCANDataset(
    TRAIN_PATH,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    shuffle=True,
    normalize_stats_path=NORMALIZE_STATS_PATH,
    fit_normalize_stats=True,  # HANYA train yang menghitung & menyimpan stats
)

val_dataset = TemporalCANDataset(
    VAL_PATH,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    shuffle=False,
    normalize_stats_path=NORMALIZE_STATS_PATH,
    fit_normalize_stats=False,  # val memuat stats yang sama dari train
)

train_tf = train_dataset.to_tf_dataset()
val_tf = val_dataset.to_tf_dataset()


# ==========================================================
# Class weight (fix bug #3 -- tidak ada penanganan imbalance)
# ==========================================================
#
# Dihitung dari distribusi label window VALID di train (bukan label
# per-frame mentah), supaya merepresentasikan persis apa yang dilihat
# model saat training.
#
class_weight = compute_class_weight_dict(train_dataset.window_labels)

print("\nClass weight (balanced):")
for cls, w in sorted(class_weight.items()):
    print(f"  kelas {cls}: {w:.4f}")


# ==========================================================
# Model
# ==========================================================

model = HybridIDS(
    d_model=4,
    num_heads=2,
    ff_dim=8,
    num_layers=1,
    gru_units=16,
    num_classes=5,
    dropout=0.1,
)


# ==========================================================
# Build model
# ==========================================================

dummy = {
    "tokens": tf.zeros(
        (1, SEQ_LEN, 10),
        dtype=tf.int32,
    ),
    "token_types": tf.zeros(
        (1, SEQ_LEN, 10),
        dtype=tf.int32,
    ),
    "positions": tf.zeros(
        (1, SEQ_LEN, 10),
        dtype=tf.int32,
    ),
    "temporal_features": tf.zeros(
        (1, SEQ_LEN, 9),
        dtype=tf.float32,
    ),
}

_ = model(dummy)

model.summary()


# ==========================================================
# Load pretrained spatial transformer
# ==========================================================

try:
    model.spatial_encoder.load_weights(PRETRAINED_SPATIAL)

    print(f"Loaded pretrained weights: {PRETRAINED_SPATIAL}")

except Exception as e:
    print("Cannot load pretrained spatial weights:")

    print(e)


# ==========================================================
# Stage 1:
# Freeze spatial encoder
# ==========================================================

model.spatial_encoder.trainable = False

print("Spatial encoder frozen.")


# ==========================================================
# Compile
# ==========================================================
#
# Fix bug #4 -- ModelCheckpoint sebelumnya monitor val_accuracy, yang
# bias ke kelas mayoritas (model "selalu prediksi Normal" otomatis dapat
# val_accuracy ~90%). Ganti ke macro-F1 -- metric ini memberi bobot sama
# ke tiap kelas (attack minoritas tidak tenggelam oleh dominasi Normal),
# jadi checkpoint yang tersimpan benar-benar merepresentasikan model yang
# bisa mendeteksi attack, bukan model yang collapse ke mayoritas.
#
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-3,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        "accuracy",
        MacroF1Score(num_classes=5),
    ],
)


# ==========================================================
# Callback
# ==========================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/hybrid_best.keras",
        monitor="val_macro_f1",
        save_best_only=True,
        mode="max",
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=8,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
    ),
]


# ==========================================================
# Stage 1 Training
# ==========================================================

history = model.fit(
    train_tf,
    validation_data=val_tf,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
)


# ==========================================================
# Stage 2 Fine-tuning
# ==========================================================

print("\nFine-tuning entire model...")

model.spatial_encoder.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        "accuracy",
        MacroF1Score(num_classes=5),
    ],
)

history_ft = model.fit(
    train_tf,
    validation_data=val_tf,
    epochs=10,
    class_weight=class_weight,
    callbacks=callbacks,
)


# ==========================================================
# Save final model
# ==========================================================

model.save("checkpoints/hybrid_final.keras")

print("Training finished.")
