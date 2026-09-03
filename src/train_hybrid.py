import dagshub
import mlflow
import tensorflow as tf
from tensorflow import keras

from data.temporal_dataset import TemporalCANDataset
from model.hybrid_ids import HybridIDS
from utils.class_weights import compute_class_weight_dict
from utils.metrics import MacroF1Score
from utils.mlflow_utils import MlflowEpochLogger, save_run_id
from utils.params import load_params

dagshub.init(repo_owner="ayasy-el", repo_name="FedCAN-IDS", mlflow=True)

# ==========================================================
# Load params.yaml (DVC-tracked hyperparameter, bukan hardcode lagi)
# ==========================================================

dataset_params = load_params("dataset")
training_params = load_params("training.hybrid")
mlflow_params = load_params("mlflow")

model_params = dict(load_params("model.hybrid"))
SEQ_LEN = model_params.pop("seq_len")


# ==========================================================
# Configuration
# ==========================================================

TRAIN_PATH = f"{dataset_params['featured_dir']}/train.parquet"
VAL_PATH = f"{dataset_params['featured_dir']}/val.parquet"

BATCH_SIZE = training_params["batch_size"]
EPOCHS_STAGE1 = training_params["epochs_stage1"]
EPOCHS_STAGE2 = training_params["epochs_stage2"]
LEARNING_RATE_STAGE1 = training_params["learning_rate_stage1"]
LEARNING_RATE_STAGE2 = training_params["learning_rate_stage2"]

PRETRAINED_SPATIAL = "checkpoints/spatial_best.keras"
NORMALIZE_STATS_PATH = "checkpoints/temporal_norm_stats.json"

RUN_ID_PATH = "checkpoints/hybrid_mlflow_run_id.txt"


# ==========================================================
# MLflow setup
# ==========================================================

mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
mlflow.set_experiment(mlflow_params["experiment_hybrid"])


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
# Class weight
# ==========================================================
class_weight = compute_class_weight_dict(train_dataset.window_labels)

print("\nClass weight (balanced):")
for cls, w in sorted(class_weight.items()):
    print(f"  kelas {cls}: {w:.4f}")


# ==========================================================
# Model
# ==========================================================

model = HybridIDS(**model_params)


# ==========================================================
# Build model
# ==========================================================

dummy = {
    "numeric_values": tf.zeros((1, SEQ_LEN, 10), dtype=tf.float32),
    "positions": tf.zeros((1, SEQ_LEN, 10), dtype=tf.int32),
    "temporal_features": tf.zeros((1, SEQ_LEN, 9), dtype=tf.float32),
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
# Stage 1: Freeze spatial encoder
# ==========================================================

model.spatial_encoder.trainable = False
print("Spatial encoder frozen.")


# ==========================================================
# Compile
# ==========================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE1),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy", MacroF1Score(num_classes=model_params["num_classes"])],
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
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
    ),
]


# ==========================================================
# Training (dibungkus 1 MLflow run -- stage 1 & stage 2 sama-sama masuk
# run yang sama, dibedakan lewat step/epoch offset supaya kurvanya
# nyambung di MLflow UI, bukan tumpang tindih dari epoch/step 0 lagi)
# ==========================================================

with mlflow.start_run() as run:
    # Simpan run_id supaya eval_hybrid.py bisa log metric test ke run
    # MLflow yang SAMA (bukan run terpisah) -- gampang dibandingkan di UI.
    save_run_id(run.info.run_id, RUN_ID_PATH)

    mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
    mlflow.log_param("model.seq_len", SEQ_LEN)
    mlflow.log_params({f"training.{k}": v for k, v in training_params.items()})

    # ------------------------------------------------------
    # Stage 1 Training
    # ------------------------------------------------------

    history = model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=EPOCHS_STAGE1,
        class_weight=class_weight,
        callbacks=callbacks + [MlflowEpochLogger()],
    )

    _stage1_epochs_ran = len(history.history.get("loss", []))

    # ------------------------------------------------------
    # Stage 2 Fine-tuning
    # ------------------------------------------------------

    print("\nFine-tuning entire model...")

    model.spatial_encoder.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", MacroF1Score(num_classes=model_params["num_classes"])],
    )

    # step_offset -- supaya kurva epoch stage 2 nyambung mulus dari
    # titik terakhir stage 1 di grafik MLflow, bukan restart dari 0.
    stage2_epoch_logger = MlflowEpochLogger(step_offset=_stage1_epochs_ran)

    history_ft = model.fit(
        train_tf,
        validation_data=val_tf,
        epochs=EPOCHS_STAGE2,
        class_weight=class_weight,
        callbacks=callbacks + [stage2_epoch_logger],
    )

    # ------------------------------------------------------
    # Save final model
    # ------------------------------------------------------

    model.save("checkpoints/hybrid_final.keras")

    mlflow.log_artifact("checkpoints/hybrid_best.keras")
    mlflow.log_artifact("checkpoints/hybrid_final.keras")
    mlflow.log_artifact(NORMALIZE_STATS_PATH)

print("Training finished.")
