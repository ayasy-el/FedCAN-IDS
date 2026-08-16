"""
Utility terpusat untuk menghitung class weight dari label training.

CATATAN SCOPE: modul ini disiapkan sebagai fondasi untuk perbaikan
"Bug #3 -- tidak ada penanganan class imbalance saat training" yang
dibahas terpisah. BELUM dipanggil dari train_spatial.py / train_hybrid.py
-- supaya train_spatial.py dan train_hybrid.py tidak diam-diam berubah
perilaku saat fokus perbaikan masih di bug #1/#2/#6.

Cara pakai nanti (saat bug #3 dikerjakan):

    from utils.class_weights import compute_class_weight_dict

    class_weight = compute_class_weight_dict(train_dataset.labels)
    model.fit(..., class_weight=class_weight)
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weight_dict(labels: np.ndarray) -> dict:
    """
    Hitung class weight "balanced" dari array label integer.

    Parameters
    ----------
    labels : np.ndarray
        Array 1D label kelas (mis. train_dataset.labels / train_dataset.y).

    Returns
    -------
    dict
        Mapping {class_index: weight}, siap dipakai sebagai argumen
        `class_weight` di `model.fit(...)`.
    """
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    return dict(zip(classes.tolist(), weights.tolist()))
