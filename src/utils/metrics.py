"""
Custom metric untuk training model streaming.

Metric di modul ini dipakai oleh model MLP dan streaming supaya nama serta
definisi metric train/validation/test konsisten.

Cara pakai nanti (saat bug #4 dikerjakan):

    from utils.metrics import MacroF1Score, MacroPrecision, MacroRecall

    model.compile(..., metrics=["accuracy", MacroPrecision(5),
                                MacroRecall(5), MacroF1Score(5)])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/streaming_best.keras",
            monitor="val_f1_macro",
            mode="max",
            save_best_only=True,
        ),
    ]
"""

import tensorflow as tf
from tensorflow import keras


class MacroF1Score(keras.metrics.Metric):
    """
    Macro-averaged F1 score untuk klasifikasi multi-kelas.

    Berbeda dari accuracy, metric ini tidak bias ke kelas mayoritas --
    kelas minoritas (attack) diberi bobot yang sama dengan kelas mayoritas
    (Normal) saat dirata-ratakan, sehingga lebih representatif untuk
    memilih checkpoint terbaik pada dataset yang imbalanced.
    """

    def __init__(self, num_classes: int, name="f1_macro", **kwargs):
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes

        self.true_positives = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", shape=(num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), tf.int32)

        y_true_oh = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_oh = tf.one_hot(y_pred, depth=self.num_classes)

        if sample_weight is not None:
            weight = tf.cast(tf.reshape(sample_weight, [-1, 1]), y_true_oh.dtype)
            y_true_oh *= weight
            y_pred_oh *= weight

        tp = tf.reduce_sum(y_true_oh * y_pred_oh, axis=0)
        fp = tf.reduce_sum((1 - y_true_oh) * y_pred_oh, axis=0)
        fn = tf.reduce_sum(y_true_oh * (1 - y_pred_oh), axis=0)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (
            self.true_positives + self.false_positives + 1e-9
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + 1e-9
        )
        f1_per_class = 2 * precision * recall / (precision + recall + 1e-9)

        return tf.reduce_mean(f1_per_class)

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


class _MacroPrecisionRecall(keras.metrics.Metric):
    """Base metric untuk macro precision/recall berbasis confusion counts."""

    result_kind = "precision"

    def __init__(self, num_classes: int, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", shape=(num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=self.num_classes)
        y_pred_oh = tf.one_hot(y_pred, depth=self.num_classes)

        if sample_weight is not None:
            weight = tf.cast(tf.reshape(sample_weight, [-1, 1]), y_true_oh.dtype)
            y_true_oh *= weight
            y_pred_oh *= weight

        self.true_positives.assign_add(tf.reduce_sum(y_true_oh * y_pred_oh, axis=0))
        self.false_positives.assign_add(
            tf.reduce_sum((1 - y_true_oh) * y_pred_oh, axis=0)
        )
        self.false_negatives.assign_add(
            tf.reduce_sum(y_true_oh * (1 - y_pred_oh), axis=0)
        )

    def result(self):
        if self.result_kind == "precision":
            values = self.true_positives / (
                self.true_positives + self.false_positives + 1e-9
            )
        else:
            values = self.true_positives / (
                self.true_positives + self.false_negatives + 1e-9
            )
        return tf.reduce_mean(values)

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


class MacroPrecision(_MacroPrecisionRecall):
    """Macro-averaged precision untuk klasifikasi multi-kelas."""

    result_kind = "precision"

    def __init__(self, num_classes: int, name="precision_macro", **kwargs):
        super().__init__(num_classes, name=name, **kwargs)


class MacroRecall(_MacroPrecisionRecall):
    """Macro-averaged recall untuk klasifikasi multi-kelas."""

    result_kind = "recall"

    def __init__(self, num_classes: int, name="recall_macro", **kwargs):
        super().__init__(num_classes, name=name, **kwargs)
