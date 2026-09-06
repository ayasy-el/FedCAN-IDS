"""Single-frame causal CAN IDS with compressed global and same-ID KV memory."""

import tensorflow as tf
from keras import layers
from tensorflow import keras

from utils.metrics import MacroF1Score


class StreamingCANIDS(keras.Model):
    def __init__(
        self,
        d_model=32,
        id_embedding_dim=8,
        numeric_dim=11,
        numeric_projection_dim=24,
        d_qk=8,
        d_v=8,
        global_memory=64,
        same_id_memory=16,
        num_ids=2048,
        num_classes=5,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model, self.d_qk, self.d_v = d_model, d_qk, d_v
        self.num_classes = num_classes
        self.global_memory, self.same_id_memory, self.num_ids = (
            global_memory,
            same_id_memory,
            num_ids,
        )
        self.id_embedding = layers.Embedding(
            num_ids, id_embedding_dim, name="can_id_embedding"
        )
        self.numeric_projection = layers.Dense(
            numeric_projection_dim, activation="gelu", name="numeric_projection"
        )
        self.frame_norm = layers.LayerNormalization(epsilon=1e-6)
        self.q_projection = layers.Dense(d_qk, use_bias=False, name="shared_q")
        self.k_projection = layers.Dense(d_qk, use_bias=False, name="shared_k")
        self.v_projection = layers.Dense(d_v, use_bias=False, name="shared_v")
        self.global_out = layers.Dense(d_model, name="global_context")
        self.same_id_out = layers.Dense(d_model, name="same_id_context")
        self.gate = layers.Dense(d_model, activation="sigmoid", name="context_gate")
        self.classifier = keras.Sequential(
            [
                layers.Dense(d_model, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(num_classes),
            ],
            name="classifier",
        )
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.macro_f1_tracker = MacroF1Score(num_classes)
        self._state_variables = None
        self._last_stream = None

    @property
    def metrics(self):
        metrics = [self.loss_tracker, self.accuracy_tracker]
        if self.macro_f1_tracker is not None:
            metrics.append(self.macro_f1_tracker)
        return metrics

    def initialize_stream_state(self, batch_size=1):
        """Allocate persistent state used by Keras ``fit``."""
        state = self.initial_state(batch_size)
        self._state_variables = [
            tf.Variable(value, trainable=False, name=f"stream_state_{i}")
            for i, value in enumerate(state)
        ]
        self._last_stream = tf.Variable(tf.fill([batch_size], b""), trainable=False, dtype=tf.string, name="last_stream")

    def reset_stream_state(self):
        if self._state_variables is not None:
            batch_size = self._state_variables[0].shape[0]
            for variable, value in zip(self._state_variables, self.initial_state(batch_size)):
                variable.assign(value)
            self._last_stream.assign(tf.fill([batch_size], b""))

    def _persistent_state(self):
        return tuple(variable.value() for variable in self._state_variables)

    def _store_stream_state(self, state, stream_id):
        for variable, value in zip(self._state_variables, state):
            variable.assign(tf.stop_gradient(value))
        self._last_stream.assign(stream_id)

    def _reset_if_new_stream(self, stream_id):
        is_new = tf.not_equal(stream_id, self._last_stream)
        self._reset_state_in_graph(is_new)

    def _reset_state_in_graph(self, is_new):
        batch_size = self._state_variables[0].shape[0]
        for variable, value in zip(self._state_variables, self.initial_state(batch_size)):
            mask = tf.reshape(is_new, [batch_size] + [1] * (len(variable.shape) - 1))
            variable.assign(tf.where(mask, value, variable))
        self._last_stream.assign(tf.where(is_new, tf.fill(tf.shape(is_new), b""), self._last_stream))

    def _masked_loss(self, labels, logits, valid_mask):
        losses = keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        valid_mask = tf.cast(valid_mask, losses.dtype)
        return tf.reduce_sum(losses * valid_mask) / tf.maximum(tf.reduce_sum(valid_mask), 1.0)

    def train_step(self, data):
        inputs, labels = data
        stream_id = inputs["stream_id"]
        self._reset_if_new_stream(stream_id)
        with tf.GradientTape() as tape:
            logits, state = self.run_sequence(
                inputs["can_id"], inputs["numeric"], self._persistent_state(),
                training=True, active_mask=inputs["valid_mask"]
            )
            loss = self._masked_loss(labels, logits, inputs["valid_mask"])
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self._store_stream_state(state, stream_id)
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits, sample_weight=inputs["valid_mask"])
        self.macro_f1_tracker.update_state(labels, logits, sample_weight=inputs["valid_mask"])
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        inputs, labels = data
        stream_id = inputs["stream_id"]
        self._reset_if_new_stream(stream_id)
        logits, state = self.run_sequence(
            inputs["can_id"], inputs["numeric"], self._persistent_state(),
            training=False, active_mask=inputs["valid_mask"]
        )
        self._store_stream_state(state, stream_id)
        loss = self._masked_loss(labels, logits, inputs["valid_mask"])
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits, sample_weight=inputs["valid_mask"])
        self.macro_f1_tracker.update_state(labels, logits, sample_weight=inputs["valid_mask"])
        return {metric.name: metric.result() for metric in self.metrics}

    def predict_step(self, data):
        """State-aware Keras ``predict`` for chronological chunks."""
        inputs = data[0] if isinstance(data, (tuple, list)) else data
        self._reset_if_new_stream(inputs["stream_id"])
        logits, state = self.run_sequence(
            inputs["can_id"], inputs["numeric"], self._persistent_state(),
            training=False, active_mask=inputs["valid_mask"]
        )
        self._store_stream_state(state, inputs["stream_id"])
        return logits

    def encode_frame(self, can_id, numeric, training=None):
        can_id = tf.clip_by_value(tf.cast(can_id, tf.int32), 0, self.num_ids - 1)
        return self.frame_norm(
            tf.concat(
                [self.id_embedding(can_id), self.numeric_projection(numeric)], axis=-1
            )
        )

    def _attention(self, q, keys, values, valid):
        scores = tf.einsum("bd,bmd->bm", q, keys) / tf.sqrt(
            tf.cast(self.d_qk, tf.float32)
        )
        scores = tf.where(
            valid, scores, tf.fill(tf.shape(scores), tf.cast(-1e9, scores.dtype))
        )
        weights = tf.nn.softmax(scores, axis=-1)
        weights = tf.where(
            tf.reduce_any(valid, axis=-1, keepdims=True),
            weights,
            tf.zeros_like(weights),
        )
        return tf.einsum("bm,bmv->bv", weights, values)

    def _step(self, can_id, numeric, state, training=None, active=None):
        gk, gv, gm, ik, iv, im = state
        can_id = tf.clip_by_value(tf.cast(can_id, tf.int32), 0, self.num_ids - 1)
        h = self.encode_frame(can_id, numeric, training)
        q = self.q_projection(h)
        g = self.global_out(self._attention(q, gk, gv, gm))
        same_k, same_v, same_m = (
            tf.gather(ik, can_id, batch_dims=1),
            tf.gather(iv, can_id, batch_dims=1),
            tf.gather(im, can_id, batch_dims=1),
        )
        s = self.same_id_out(self._attention(q, same_k, same_v, same_m))
        alpha = self.gate(tf.concat([h, g, s], axis=-1))
        logits = self.classifier(h + alpha * g + (1.0 - alpha) * s, training=training)

        # Causal invariant: this update is executed only after logits exist.
        k, v = self.k_projection(h), self.v_projection(h)
        gk = tf.concat([gk[:, 1:], k[:, None, :]], axis=1)
        gv = tf.concat([gv[:, 1:], v[:, None, :]], axis=1)
        gm = tf.concat([gm[:, 1:], tf.ones_like(gm[:, :1])], axis=1)
        b = tf.range(tf.shape(can_id)[0], dtype=tf.int32)
        indices = tf.stack([b, can_id], axis=1)
        old_k, old_v, old_m = (
            tf.gather_nd(ik, indices),
            tf.gather_nd(iv, indices),
            tf.gather_nd(im, indices),
        )
        new_k = tf.concat([old_k[:, 1:], k[:, None, :]], axis=1)
        new_v = tf.concat([old_v[:, 1:], v[:, None, :]], axis=1)
        new_m = tf.concat([old_m[:, 1:], tf.ones_like(old_m[:, :1])], axis=1)
        updated = (
            gk,
            gv,
            gm,
            tf.tensor_scatter_nd_update(ik, indices, new_k),
            tf.tensor_scatter_nd_update(iv, indices, new_v),
            tf.tensor_scatter_nd_update(im, indices, new_m),
        )
        if active is not None:
            updated = tuple(
                tf.where(
                    tf.reshape(tf.cast(active, tf.bool), [-1] + [1] * (value.shape.rank - 1)),
                    value,
                    old,
                )
                for value, old in zip(updated, state)
            )
        return logits, updated

    def initial_state(self, batch_size):
        z = tf.zeros
        return (
            z((batch_size, self.global_memory, self.d_qk)),
            z((batch_size, self.global_memory, self.d_v)),
            z((batch_size, self.global_memory), tf.bool),
            z((batch_size, self.num_ids, self.same_id_memory, self.d_qk)),
            z((batch_size, self.num_ids, self.same_id_memory, self.d_v)),
            z((batch_size, self.num_ids, self.same_id_memory), tf.bool),
        )

    def call(self, inputs, training=False):
        ids, numeric = inputs["can_id"], inputs["numeric"]
        return self.run_sequence(ids, numeric, training=training)[0]

    def run_sequence(self, ids, numeric, state=None, training=False, active_mask=None):
        """Process a chronological chunk and return ``(logits, state)``."""
        if state is None:
            state = self.initial_state(tf.shape(ids)[0])
        outputs = tf.TensorArray(tf.float32, size=tf.shape(ids)[1])

        def condition(t, state, outputs):
            return t < tf.shape(ids)[1]

        def body(t, state, outputs):
            active = None if active_mask is None else active_mask[:, t]
            logits, state = self._step(ids[:, t], numeric[:, t], state, training, active)
            return t + 1, state, outputs.write(t, logits)

        _, state, outputs = tf.while_loop(
            condition,
            body,
            (tf.constant(0), state, outputs),
        )
        return tf.transpose(outputs.stack(), [1, 0, 2]), state

    def predict_frame(self, can_id, numeric, state=None):
        """Run one frame and return ``(logits, updated_state)``."""
        if state is None:
            state = self.initial_state(tf.shape(can_id)[0])
        return self._step(can_id, numeric, state, training=False)
