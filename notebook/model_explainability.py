import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Explainability Analysis -- Hybrid IDS (CAN Bus)

    Notebook ini untuk menjawab tiga pertanyaan sebelum memutuskan langkah
    perbaikan (tuning threshold / kapasitas model / ganti metode / adjust
    feature engineering):

    1. **Distribusi tiap fitur per kelas** -- fitur mana yang benar-benar
       memisahkan kelas, fitur mana yang tumpang-tindih (tidak
       diskriminatif).
    2. **Representasi spatial transformer** -- divisualisasikan (PCA/t-SNE)
       supaya kelihatan apakah embedding-nya SUDAH cukup terpisah antar
       kelas (masalah ada di training/threshold) atau MEMANG masih
       tumpang-tindih (masalah ada di representasi/fitur/arsitektur).
    3. **Analisis kegagalan per-kelas** -- generik, bukan cuma untuk
       Replay. Ganti `target_class_selector` di bawah untuk kelas attack
       apapun, di dataset apapun -- satu-satunya bagian yang spesifik ke
       dataset ada di cell **Config** (paths, nama kelas, daftar kolom
       fitur). Bagian analisisnya sendiri dirancang dataset-agnostic.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path
    import json

    import numpy as np
    import polars as pl

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    from scipy import stats

    return Path, json, np, pl, plt, px, sns, stats, sys


@app.cell
def _():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    return PCA, TSNE


@app.cell
def _():
    import tensorflow as tf
    from tensorflow import keras

    return (tf,)


@app.cell
def _(Path, sys):
    # Notebook ini ada di hybrid_ids/notebooks/, source code ada di
    # hybrid_ids/src/ -- tambahkan src/ ke sys.path supaya bisa import
    # module project (model.spatial_transformer, data.dataset) LANGSUNG,
    # bukan menduplikasi logic tokenisasi/model secara manual. Ini penting
    # supaya embedding yang divisualisasikan di notebook 100% konsisten
    # dengan apa yang benar-benar dilihat model saat training/eval, bukan
    # rekonstruksi yang bisa saja meleset.
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = NOTEBOOK_DIR.parent
    SRC_DIR = PROJECT_ROOT / "src"

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    return (PROJECT_ROOT,)


@app.cell
def _():
    from model.spatial_transformer import SpatialTransformer

    return (SpatialTransformer,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Config -- SATU-SATUNYA bagian yang spesifik ke dataset ini
    """)
    return


@app.cell
def _(PROJECT_ROOT):
    # ---------------------------------------------------------------
    # Ganti isi cell ini kalau mau pakai notebook yang sama untuk
    # dataset lain / arsitektur lain -- cell di bawahnya (distribusi
    # fitur, embedding, confusion matrix, analisis kegagalan) tidak
    # perlu diubah sama sekali, selama kolom Class/session_id dan
    # struktur output evaluasi (classification_report JSON) konsisten.
    # ---------------------------------------------------------------

    CLASS_NAMES = [
        "Normal",
        "Flooding",
        "Fuzzing",
        "Spoofing",
        "Replay",
    ]

    # Kolom fitur temporal -- urutan HARUS sama dengan yang dipakai di
    # src/data/feature.py dan src/data/temporal_dataset.py.
    TEMPORAL_FEATURE_COLUMNS = [
        "delta_t",
        "delta_t_z",
        "jitter",
        "hamming_distance",
        "frame_rate_20ms",
        "id_rate_20ms",
        "id_entropy_20ms",
        "dominant_id_ratio_20ms",
        "bus_load_proxy_20ms",
    ]

    RAW_SPLIT_PATHS = {
        "train": PROJECT_ROOT / "data/processed/car_hacking/train.parquet",
        "val": PROJECT_ROOT / "data/processed/car_hacking/val.parquet",
        "test": PROJECT_ROOT / "data/processed/car_hacking/test.parquet",
    }

    FEATURED_SPLIT_PATHS = {
        "train": PROJECT_ROOT / "data/processed/car_hacking/featured/train.parquet",
        "val": PROJECT_ROOT / "data/processed/car_hacking/featured/val.parquet",
        "test": PROJECT_ROOT / "data/processed/car_hacking/featured/test.parquet",
    }

    # Hyperparameter arsitektur -- HARUS sama persis dengan yang dipakai
    # saat training (lihat src/train_spatial.py), supaya load_weights cocok.
    SPATIAL_MODEL_CONFIG = dict(
        d_model=4,
        num_heads=2,
        ff_dim=8,
        num_layers=1,
        num_classes=5,
        dropout=0.1,
    )

    SPATIAL_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints/spatial_best.keras"

    HYBRID_METRICS_JSON_PATH = (
        PROJECT_ROOT / "reports/metrics/hybrid_classification_report.json"
    )
    SPATIAL_METRICS_JSON_PATH = (
        PROJECT_ROOT / "reports/metrics/spatial_classification_report.json"
    )
    return (
        CLASS_NAMES,
        FEATURED_SPLIT_PATHS,
        HYBRID_METRICS_JSON_PATH,
        RAW_SPLIT_PATHS,
        SPATIAL_CHECKPOINT_PATH,
        SPATIAL_METRICS_JSON_PATH,
        SPATIAL_MODEL_CONFIG,
        TEMPORAL_FEATURE_COLUMNS,
    )


@app.cell
def _(CLASS_NAMES):
    # Palet warna kontras tinggi antar kelas -- dipilih supaya hue-nya
    # jauh satu sama lain (bukan gradasi/berdekatan seperti default
    # sebagian library), supaya titik tiap kelas gampang dibedakan mata
    # di scatter plot PCA/t-SNE maupun kurva densitas.
    _PALETTE = [
        "#1f77b4",  # biru
        "#e6194B",  # merah terang
        "#f58231",  # oranye
        "#3cb44b",  # hijau
        "#911eb4",  # ungu
        "#ffe119",  # kuning
        "#42d4f4",  # cyan
        "#f032e6",  # magenta
    ]
    CLASS_COLOR_MAP = {
        _cname: _PALETTE[_idx % len(_PALETTE)]
        for _idx, _cname in enumerate(CLASS_NAMES)
    }
    return (CLASS_COLOR_MAP,)


@app.cell
def _(mo):
    split_selector = mo.ui.dropdown(
        options=["train", "val", "test"],
        value="test",
        label="Split yang dianalisis",
    )
    split_selector
    return (split_selector,)


@app.cell
def _(FEATURED_SPLIT_PATHS, pl, split_selector):
    df_featured = pl.read_parquet(FEATURED_SPLIT_PATHS[split_selector.value])
    return (df_featured,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Distribusi Kelas per Split
    """)
    return


@app.cell
def _(CLASS_NAMES, FEATURED_SPLIT_PATHS, pl, px):
    _rows = []
    for _split_name, _path in FEATURED_SPLIT_PATHS.items():
        _df = pl.read_parquet(_path).select("Class")
        _counts = _df.group_by("Class").len().rename({"len": "count"})
        for _row in _counts.iter_rows(named=True):
            _rows.append(
                {
                    "split": _split_name,
                    "class_name": CLASS_NAMES[_row["Class"]],
                    "count": _row["count"],
                }
            )

    class_dist_df = pl.DataFrame(_rows)

    fig_class_dist = px.bar(
        class_dist_df.to_pandas(),
        x="class_name",
        y="count",
        color="split",
        barmode="group",
        log_y=True,
        title="Distribusi kelas per split (skala log -- supaya kelas minoritas tetap kelihatan)",
    )
    fig_class_dist
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Distribusi Fitur Temporal per Kelas

    Kalau boxplot suatu fitur untuk dua kelas **tumpang-tindih total**,
    fitur itu tidak berguna untuk memisahkan kedua kelas itu -- apapun
    model/arsitektur yang dipakai. Kalau *semua* fitur tumpang-tindih
    untuk kelas tertentu, itu tanda perlu fitur BARU, bukan model yang
    lebih besar.
    """)
    return


@app.cell
def _(TEMPORAL_FEATURE_COLUMNS, mo):
    feature_selector = mo.ui.dropdown(
        options=TEMPORAL_FEATURE_COLUMNS,
        value=TEMPORAL_FEATURE_COLUMNS[0],
        label="Fitur",
    )
    log_scale_toggle = mo.ui.checkbox(value=True, label="Skala log (sumbu Y)")
    mo.hstack([feature_selector, log_scale_toggle])
    return feature_selector, log_scale_toggle


@app.cell
def _(CLASS_NAMES, df_featured, feature_selector, log_scale_toggle, plt, sns):
    _fig, _ax = plt.subplots(figsize=(9, 5))

    _plot_df = df_featured.select(["Class", feature_selector.value]).to_pandas()
    _plot_df["class_name"] = _plot_df["Class"].map(lambda i: CLASS_NAMES[i])

    sns.boxplot(
        data=_plot_df,
        x="class_name",
        y=feature_selector.value,
        order=CLASS_NAMES,
        ax=_ax,
    )

    if log_scale_toggle.value:
        _ax.set_yscale("symlog")

    _ax.set_title(f"Distribusi '{feature_selector.value}' per kelas")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Ringkasan separability semua fitur (one-way ANOVA)
    """)
    return


@app.cell
def _(CLASS_NAMES, TEMPORAL_FEATURE_COLUMNS, df_featured, pl, stats):
    _rows = []

    for _feat in TEMPORAL_FEATURE_COLUMNS:
        _groups = [
            df_featured.filter(pl.col("Class") == _cls_idx)[_feat]
            .drop_nulls()
            .to_numpy()
            for _cls_idx in range(len(CLASS_NAMES))
        ]
        _groups = [g for g in _groups if len(g) > 1]

        if len(_groups) < 2:
            continue

        _f_stat, _p_value = stats.f_oneway(*_groups)

        _rows.append(
            {
                "feature": _feat,
                "f_statistic": _f_stat,
                "p_value": _p_value,
            }
        )

    separability_table = pl.DataFrame(_rows).sort("f_statistic", descending=True)

    separability_table
    return (separability_table,)


@app.cell
def _(px, separability_table):
    fig_anova = px.bar(
        separability_table.to_pandas(),
        x="f_statistic",
        y="feature",
        orientation="h",
        log_x=True,
        title="ANOVA F-statistic per fitur (skala log) -- makin tinggi, makin diskriminatif",
        labels={"f_statistic": "F-statistic (log scale)", "feature": "Fitur"},
    )
    fig_anova.update_yaxes(categoryorder="total ascending")
    fig_anova
    return


@app.cell
def _(mo):
    mo.md(r"""
    F-statistic tinggi = fitur itu SANGAT memisahkan kelas satu sama
    lain (secara rata-rata). F-statistic rendah (mendekati fitur paling
    bawah tabel) = fitur itu hampir tidak berguna untuk membedakan
    kelas -- kalau fitur yang relevan untuk kelas yang gagal (mis.
    `hamming_distance` untuk Replay) F-statistic-nya rendah, itu
    indikasi kuat fiturnya sendiri memang belum cukup menangkap pola
    serangan itu.

    **Catatan:** F-statistic ini one-way ANOVA lintas SEMUA kelas
    sekaligus -- fitur bisa saja F-statistic tinggi karena berhasil
    memisahkan kelas lain, padahal tetap tumpang-tindih untuk 2 kelas
    spesifik (mis. Replay vs Normal). Untuk itu, lihat bagian
    **Analisis Kegagalan Per-Kelas** di bawah, yang membandingkan
    1 kelas spesifik vs Normal secara langsung.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Ilustrasi Within-group vs Between-group Variance (kurva densitas per kelas)

    Kurva densitas (KDE) tiap kelas untuk fitur yang dipilih di atas --
    versi "data asli" dari ilustrasi konsep ANOVA: kalau kurva antar
    kelas saling menjauh dan tidak tumpang-tindih, artinya
    *between-group variance* besar relatif terhadap *within-group
    variance* -- fitur ini diskriminatif untuk kelas-kelas tersebut.
    Sebaliknya, kalau kurva-kurva saling menumpuk, fitur ini tidak
    banyak membantu memisahkan kelas-kelas itu.
    """)
    return


@app.cell
def _(CLASS_COLOR_MAP, CLASS_NAMES, df_featured, feature_selector, pl, plt, sns):
    _fig, _ax = plt.subplots(figsize=(9, 5))

    for _cls_idx, _cname in enumerate(CLASS_NAMES):
        _vals = (
            df_featured.filter(pl.col("Class") == _cls_idx)[feature_selector.value]
            .drop_nulls()
            .to_numpy()
        )
        if len(_vals) > 1:
            sns.kdeplot(
                _vals,
                fill=True,
                alpha=0.45,
                linewidth=1.5,
                color=CLASS_COLOR_MAP[_cname],
                label=_cname,
                ax=_ax,
            )

    _ax.set_title(f"Distribusi densitas '{feature_selector.value}' per kelas")
    _ax.set_xlabel(feature_selector.value)
    _ax.set_ylabel("Density")
    _ax.legend(fontsize=8, frameon=False)
    for _spine in ("top", "right"):
        _ax.spines[_spine].set_visible(False)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Visualisasi Representasi Spatial (Transformer Embedding)

    Sample beberapa baris per kelas dari split RAW (bukan featured --
    spatial model cuma butuh token per-frame), jalankan lewat spatial
    transformer yang sudah di-training, ambil `z_spatial` (representasi
    sebelum classifier head), lalu reduksi ke 2D untuk divisualisasikan.

    **Cara baca:**
    - Kalau titik-titik per kelas membentuk cluster yang **terpisah
      jelas** -> representasi spatial SUDAH cukup, masalah performa
      kemungkinan di training (class_weight/threshold/monitor metric).
    - Kalau titik-titik **bertumpuk total** dengan kelas lain (terutama
      Normal) -> representasi spatial belum cukup untuk kelas itu --
      pertimbangkan fitur tambahan atau naikkan kapasitas model
      (`d_model`), BUKAN cuma tuning threshold.

    Bagian **3A** di bawah menampilkan baseline PCA/t-SNE pada fitur
    RAW (Arbitration_ID, DLC, Data_0..Data_7) -- TANPA melalui model
    apapun -- supaya bisa dibandingkan langsung dengan embedding
    spatial transformer di bagian **3B**.
    """)
    return


@app.cell
def _(mo):
    embed_sample_size = mo.ui.slider(
        start=200,
        stop=5000,
        step=200,
        value=1500,
        label="Sample per kelas",
    )
    embed_method = mo.ui.dropdown(
        options=["PCA", "t-SNE"],
        value="PCA",
        label="Metode reduksi dimensi",
    )
    embed_dims = mo.ui.dropdown(
        options=["2D", "3D"],
        value="2D",
        label="Dimensi plot",
    )
    mo.hstack([embed_sample_size, embed_method, embed_dims])
    return embed_dims, embed_method, embed_sample_size


@app.cell
def _(RAW_SPLIT_PATHS, pl, split_selector):
    raw_df = pl.read_parquet(RAW_SPLIT_PATHS[split_selector.value])
    return (raw_df,)


@app.cell
def _(CLASS_NAMES, embed_sample_size, pl, raw_df):
    _parts = []

    for _cls_idx in range(len(CLASS_NAMES)):
        _subset = raw_df.filter(pl.col("Class") == _cls_idx)
        _n = min(embed_sample_size.value, _subset.height)

        if _n == 0:
            continue

        _parts.append(_subset.sample(n=_n, seed=42, with_replacement=False))

    sample_df = pl.concat(_parts)
    return (sample_df,)


@app.cell
def _(np, pl, sample_df):
    # Tokenisasi -- MENIRU PERSIS logic di src/data/dataset.py
    # (CANDataset.prepare), termasuk konvensi token_types=3 untuk byte PAD,
    # supaya embedding yang dihasilkan konsisten dengan apa yang dilihat
    # model saat training/eval spatial (bukan reimplementasi yang bisa
    # meleset). Diterapkan cuma ke sample kecil (bukan seluruh split) biar
    # notebook tetap responsif untuk eksplorasi interaktif.

    _df = sample_df.with_columns(
        pl.col("Arbitration_ID")
        .str.to_integer(base=16, strict=False)
        .fill_null(0)
        .cast(pl.UInt32)
        .alias("Arbitration_ID")
    )

    _payload_exprs = []
    for _i in range(8):
        _c = f"Data_{_i}"
        _payload_exprs.append(
            pl.col(_c)
            .str.to_integer(base=16, strict=False)
            .fill_null(256)
            .cast(pl.UInt16)
            .alias(_c)
        )
    _df = _df.with_columns(_payload_exprs)

    tokens_np = np.column_stack(
        [
            _df["Arbitration_ID"].to_numpy().astype(np.int32),
            _df["DLC"].to_numpy().astype(np.int32),
            *[_df[f"Data_{_i}"].to_numpy().astype(np.int32) for _i in range(8)],
        ]
    )

    _n_rows = len(_df)

    token_types_np = np.full((_n_rows, 10), 2, dtype=np.int32)
    token_types_np[:, 0] = 0
    token_types_np[:, 1] = 1
    _payload = tokens_np[:, 2:]
    token_types_np[:, 2:] = np.where(_payload == 256, 3, 2)

    positions_np = np.broadcast_to(np.arange(10, dtype=np.int32), (_n_rows, 10)).copy()

    labels_np = _df["Class"].to_numpy().astype(np.int32)
    return labels_np, positions_np, token_types_np, tokens_np


@app.cell
def _(mo):
    mo.md(r"""
    ### 3A. Baseline -- PCA/t-SNE pada RAW DATA (ID, DLC, Byte0-7)

    Reduksi dimensi memakai fitur mentah yang PERSIS sama yang dipakai
    sebagai input token spatial transformer (Arbitration_ID, DLC,
    Data_0..Data_7 dalam bentuk integer -- byte kosong diisi nilai `256`
    sebagai flag padding), tapi TANPA melewati model apapun. Ini
    baseline pembanding: kalau cluster pada baseline raw data ini SUDAH
    terpisah sebaik (atau lebih baik dari) embedding spatial transformer
    di bagian 3B, berarti transformer belum menambah nilai representasi
    apapun dibanding fitur mentahnya.
    """)
    return


@app.cell
def _(tokens_np):
    # Baseline "tanpa model" -- pakai fitur mentah apa adanya, tanpa
    # melalui SpatialTransformer, supaya jadi pembanding representasi.
    raw_features_np = tokens_np.astype(float)
    return (raw_features_np,)


@app.cell
def _(PCA, TSNE, embed_dims, embed_method, mo, raw_features_np):
    _n_components = 3 if embed_dims.value == "3D" else 2

    if embed_method.value == "PCA":
        _reducer = PCA(n_components=_n_components, random_state=42)
        raw_reduced = _reducer.fit_transform(raw_features_np)
        _total_var = _reducer.explained_variance_ratio_.sum()
        raw_variance_note = (
            f"PCA {embed_dims.value} pada **raw data** menjelaskan "
            f"**{_total_var:.1%}** varians fitur mentah asli."
        )
    else:
        _perplexity = min(30, max(5, len(raw_features_np) // 100))
        _reducer = TSNE(
            n_components=_n_components,
            init="pca",
            perplexity=_perplexity,
            random_state=42,
        )
        raw_reduced = _reducer.fit_transform(raw_features_np)
        raw_variance_note = (
            "**Catatan t-SNE (raw data):** jarak ANTAR cluster TIDAK "
            "bermakna secara global (cuma struktur tetangga lokal yang "
            "valid)."
        )

    mo.md(raw_variance_note)
    return (raw_reduced,)


@app.cell
def _(
    CLASS_COLOR_MAP,
    CLASS_NAMES,
    embed_dims,
    embed_method,
    labels_np,
    pl,
    px,
    raw_reduced,
):
    _cols = {"dim1": raw_reduced[:, 0], "dim2": raw_reduced[:, 1]}
    if embed_dims.value == "3D":
        _cols["dim3"] = raw_reduced[:, 2]

    _plot_df = pl.DataFrame(
        {
            **_cols,
            "class_name": [CLASS_NAMES[c] for c in labels_np],
        }
    ).to_pandas()

    _title = f"Raw data embedding ({embed_method.value}, {embed_dims.value}) -- baseline TANPA model"

    if embed_dims.value == "3D":
        fig_embedding_raw = px.scatter_3d(
            _plot_df,
            x="dim1",
            y="dim2",
            z="dim3",
            color="class_name",
            color_discrete_map=CLASS_COLOR_MAP,
            opacity=0.55,
            title=_title,
            category_orders={"class_name": CLASS_NAMES},
        )
        fig_embedding_raw.update_traces(marker=dict(size=3))
    else:
        fig_embedding_raw = px.scatter(
            _plot_df,
            x="dim1",
            y="dim2",
            color="class_name",
            color_discrete_map=CLASS_COLOR_MAP,
            opacity=0.55,
            title=_title,
            category_orders={"class_name": CLASS_NAMES},
        )

    fig_embedding_raw
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3B. Embedding Spatial Transformer (setelah model)
    """)
    return


@app.cell
def _(
    SPATIAL_CHECKPOINT_PATH,
    SPATIAL_MODEL_CONFIG,
    SpatialTransformer,
    mo,
    tf,
):
    _spatial_model = SpatialTransformer(**SPATIAL_MODEL_CONFIG)

    _dummy = {
        "tokens": tf.zeros((1, 10), dtype=tf.int32),
        "token_types": tf.zeros((1, 10), dtype=tf.int32),
        "positions": tf.zeros((1, 10), dtype=tf.int32),
    }
    _ = _spatial_model(_dummy)

    if SPATIAL_CHECKPOINT_PATH.exists():
        _spatial_model.load_weights(str(SPATIAL_CHECKPOINT_PATH))
        spatial_model = _spatial_model
        _ = mo.md(f"Checkpoint dimuat: `{SPATIAL_CHECKPOINT_PATH}`")
    else:
        spatial_model = None
        _ = mo.md(
            f"**Checkpoint tidak ditemukan:** `{SPATIAL_CHECKPOINT_PATH}` -- "
            f"jalankan `src/train_spatial.py` dulu supaya bagian embedding "
            f"di bawah bisa jalan."
        )
    return (spatial_model,)


@app.cell
def _(mo, positions_np, spatial_model, tf, token_types_np, tokens_np):
    if spatial_model is None:
        z_embeddings = None
        _ = mo.md("Skip: spatial_model belum ada.")
    else:
        _inputs = {
            "tokens": tf.constant(tokens_np, dtype=tf.int32),
            "token_types": tf.constant(token_types_np, dtype=tf.int32),
            "positions": tf.constant(positions_np, dtype=tf.int32),
        }
        z_embeddings = spatial_model(
            _inputs, training=False, return_embedding=True
        ).numpy()
    return (z_embeddings,)


@app.cell
def _(PCA, TSNE, embed_dims, embed_method, mo, z_embeddings):
    if z_embeddings is None:
        z_reduced = None
        explained_variance_note = ""
    else:
        _n_components = 3 if embed_dims.value == "3D" else 2

        if embed_method.value == "PCA":
            _reducer = PCA(n_components=_n_components, random_state=42)
            z_reduced = _reducer.fit_transform(z_embeddings)
            _total_var = _reducer.explained_variance_ratio_.sum()
            explained_variance_note = (
                f"PCA {embed_dims.value} menjelaskan **{_total_var:.1%}** "
                f"varians embedding asli. Kalau angka ini rendah (<50%), "
                f"pertimbangkan ganti ke t-SNE -- PCA linear mungkin tidak "
                f"menangkap struktur cluster yang sebenarnya ada."
            )
        else:
            _perplexity = min(30, max(5, len(z_embeddings) // 100))
            _reducer = TSNE(
                n_components=_n_components,
                init="pca",
                perplexity=_perplexity,
                random_state=42,
            )
            z_reduced = _reducer.fit_transform(z_embeddings)
            explained_variance_note = (
                "**Catatan t-SNE:** jarak ANTAR cluster di plot t-SNE TIDAK "
                "bermakna secara global (cuma struktur tetangga lokal yang "
                "valid). Jangan simpulkan 'cluster A lebih dekat ke B "
                "daripada C' dari jarak visual t-SNE."
            )

    mo.md(explained_variance_note)
    return (z_reduced,)


@app.cell
def _(
    CLASS_COLOR_MAP,
    CLASS_NAMES,
    embed_dims,
    embed_method,
    labels_np,
    pl,
    px,
    z_reduced,
):
    if z_reduced is None:
        _fig_embedding = None
    else:
        _cols = {"dim1": z_reduced[:, 0], "dim2": z_reduced[:, 1]}
        if embed_dims.value == "3D":
            _cols["dim3"] = z_reduced[:, 2]

        _plot_df = pl.DataFrame(
            {
                **_cols,
                "class_name": [CLASS_NAMES[c] for c in labels_np],
            }
        ).to_pandas()

        _title = f"Spatial embedding ({embed_method.value}, {embed_dims.value})"

        if embed_dims.value == "3D":
            _fig_embedding = px.scatter_3d(
                _plot_df,
                x="dim1",
                y="dim2",
                z="dim3",
                color="class_name",
                color_discrete_map=CLASS_COLOR_MAP,
                opacity=0.55,
                title=_title,
                category_orders={"class_name": CLASS_NAMES},
            )
            _fig_embedding.update_traces(marker=dict(size=3))
        else:
            _fig_embedding = px.scatter(
                _plot_df,
                x="dim1",
                y="dim2",
                color="class_name",
                color_discrete_map=CLASS_COLOR_MAP,
                opacity=0.55,
                title=_title,
                category_orders={"class_name": CLASS_NAMES},
            )

    _fig_embedding
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3B. Message Timing Plot (mereplikasi gaya plot di paper referensi)

    Scatter fitur terhadap waktu (per `session_id`), dengan overlay
    warna Normal vs Attack -- mereplikasi gaya "Message Timing" /
    "Δt vs Elapsed Time" yang umum dipakai di paper CAN IDS untuk
    menunjukkan SECARA VISUAL kapan & seberapa jelas serangan mengubah
    pola suatu fitur dari waktu ke waktu.

    - Pilih **Arbitration_ID** untuk gaya plot seperti "Message Timing"
      (ID pesan vs waktu).
    - Pilih salah satu fitur temporal (mis. `delta_t`) untuk gaya plot
      seperti "Δt vs Elapsed Time".
    """)
    return


@app.cell
def _(TEMPORAL_FEATURE_COLUMNS, df_featured, mo):
    _session_options = df_featured["session_id"].unique().sort().to_list()
    timing_session_selector = mo.ui.dropdown(
        options=_session_options,
        value=_session_options[0] if _session_options else None,
        label="session_id",
    )
    timing_feature_selector = mo.ui.dropdown(
        options=["Arbitration_ID"] + TEMPORAL_FEATURE_COLUMNS,
        value="delta_t",
        label="Fitur (sumbu Y)",
    )
    timing_color_mode = mo.ui.dropdown(
        options=["Binary (Normal vs Attack)", "Per kelas"],
        value="Binary (Normal vs Attack)",
        label="Mode warna",
    )
    mo.hstack([timing_session_selector, timing_feature_selector, timing_color_mode])
    return timing_color_mode, timing_feature_selector, timing_session_selector


@app.cell
def _(
    CLASS_COLOR_MAP,
    CLASS_NAMES,
    df_featured,
    pl,
    plt,
    timing_color_mode,
    timing_feature_selector,
    timing_session_selector,
):
    _session_df = df_featured.filter(
        pl.col("session_id") == timing_session_selector.value
    ).sort("Timestamp")

    _t0 = _session_df["Timestamp"].min()
    _elapsed = (_session_df["Timestamp"] - _t0).to_numpy()

    if timing_feature_selector.value == "Arbitration_ID":
        # Arbitration_ID masih hex string ("4F1") di data hasil feature
        # engineering (konversi ke integer baru terjadi saat tokenisasi
        # untuk model) -- konversi dulu supaya bisa diplot sebagai sumbu Y
        # numerik, mirip gaya "Message Timing" di paper.
        _y = (
            _session_df["Arbitration_ID"]
            .str.to_integer(base=16, strict=False)
            .fill_null(0)
            .to_numpy()
        )
    else:
        _y = _session_df[timing_feature_selector.value].to_numpy()

    _classes = _session_df["Class"].to_numpy()

    _fig, _ax = plt.subplots(figsize=(11, 4))

    if timing_color_mode.value == "Binary (Normal vs Attack)":
        _is_attack = _classes != 0
        _ax.scatter(
            _elapsed[~_is_attack],
            _y[~_is_attack],
            s=4,
            color=CLASS_COLOR_MAP["Normal"],
            alpha=0.5,
            label="Normal",
        )
        _ax.scatter(
            _elapsed[_is_attack],
            _y[_is_attack],
            s=4,
            color="#e6194B",
            alpha=0.7,
            label="Attack",
        )
    else:
        for _idx, _cname in enumerate(CLASS_NAMES):
            _mask = _classes == _idx
            if _mask.sum() == 0:
                continue
            _ax.scatter(
                _elapsed[_mask],
                _y[_mask],
                s=4,
                alpha=0.6,
                color=CLASS_COLOR_MAP[_cname],
                label=_cname,
            )

    _ax.set_xlabel("Elapsed Time (s)")
    _ax.set_ylabel(timing_feature_selector.value)
    _ax.set_title(
        f"{timing_feature_selector.value} vs waktu -- session "
        f"'{timing_session_selector.value}'"
    )
    _ax.legend(markerscale=3, fontsize=8, loc="upper right")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Confusion Matrix & Classification Report (Model Saat Ini)
    """)
    return


@app.cell
def _(HYBRID_METRICS_JSON_PATH, SPATIAL_METRICS_JSON_PATH, json, mo):
    def _load_report(path):
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    hybrid_report = _load_report(HYBRID_METRICS_JSON_PATH)
    spatial_report = _load_report(SPATIAL_METRICS_JSON_PATH)

    if hybrid_report is None:
        _ = mo.md(
            f"**Belum ada report:** `{HYBRID_METRICS_JSON_PATH}` -- jalankan "
            f"`src/eval_hybrid.py` dulu."
        )
    return (hybrid_report,)


@app.cell
def _(CLASS_NAMES, hybrid_report, mo, np, plt, sns):
    if hybrid_report is None:
        fig_cm = None
    else:
        _cm = np.array(hybrid_report["confusion_matrix"])
        _fig, _ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            _cm,
            annot=True,
            fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cmap="Blues",
            ax=_ax,
        )
        _ax.set_xlabel("Predicted")
        _ax.set_ylabel("True")
        _ax.set_title("Confusion Matrix -- Hybrid IDS (model saat ini)")
        plt.tight_layout()
        fig_cm = _fig

    fig_cm if fig_cm is not None else mo.md("(confusion matrix belum tersedia)")
    return


@app.cell
def _(CLASS_NAMES, hybrid_report, pl, px):
    if hybrid_report is None:
        fig_per_class = None
    else:
        _rows = []
        for _cname in CLASS_NAMES:
            _metrics = hybrid_report["classification_report"][_cname]
            for _metric_name in ["precision", "recall", "f1-score"]:
                _rows.append(
                    {
                        "class_name": _cname,
                        "metric": _metric_name,
                        "value": _metrics[_metric_name],
                    }
                )

        _df = pl.DataFrame(_rows).to_pandas()

        fig_per_class = px.bar(
            _df,
            x="class_name",
            y="value",
            color="metric",
            barmode="group",
            range_y=[0, 1],
            title="Precision / Recall / F1 per kelas -- Hybrid IDS",
            category_orders={"class_name": CLASS_NAMES},
        )

    fig_per_class
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Analisis Kegagalan Per-Kelas (generik -- ganti kelas apapun)

    Bandingkan distribusi tiap fitur untuk 1 kelas target vs Normal
    secara langsung (bukan ANOVA lintas semua kelas seperti di atas) --
    ini yang paling relevan untuk kelas yang gagal karena tumpang-tindih
    SPESIFIK dengan Normal (seperti Replay), bukan tumpang-tindih dengan
    semua kelas.
    """)
    return


@app.cell
def _(CLASS_NAMES, mo):
    target_class_selector = mo.ui.dropdown(
        options=CLASS_NAMES,
        value="Replay" if "Replay" in CLASS_NAMES else CLASS_NAMES[-1],
        label="Kelas target (yang mau dianalisis kegagalannya)",
    )
    target_class_selector
    return (target_class_selector,)


@app.cell
def _(
    CLASS_COLOR_MAP,
    CLASS_NAMES,
    TEMPORAL_FEATURE_COLUMNS,
    df_featured,
    pl,
    plt,
    target_class_selector,
):
    _target_idx = CLASS_NAMES.index(target_class_selector.value)
    _normal_idx = CLASS_NAMES.index("Normal") if "Normal" in CLASS_NAMES else 0

    _n_features = len(TEMPORAL_FEATURE_COLUMNS)
    _n_cols = 3
    _n_rows_grid = (_n_features + _n_cols - 1) // _n_cols

    _fig, _axes = plt.subplots(
        _n_rows_grid, _n_cols, figsize=(4.5 * _n_cols, 3.5 * _n_rows_grid)
    )
    _axes = _axes.flatten()

    _target_df = df_featured.filter(pl.col("Class") == _target_idx)
    _normal_df = df_featured.filter(pl.col("Class") == _normal_idx)

    for _idx, _feat in enumerate(TEMPORAL_FEATURE_COLUMNS):
        _ax = _axes[_idx]

        _target_vals = _target_df[_feat].drop_nulls().to_numpy()
        _normal_vals = _normal_df[_feat].drop_nulls().to_numpy()

        _ax.hist(
            _normal_vals,
            bins=40,
            alpha=0.5,
            label="Normal",
            density=True,
            color=CLASS_COLOR_MAP["Normal"],
        )
        _ax.hist(
            _target_vals,
            bins=40,
            alpha=0.5,
            label=target_class_selector.value,
            density=True,
            color=CLASS_COLOR_MAP[target_class_selector.value],
        )
        _ax.set_title(_feat, fontsize=10)
        _ax.legend(fontsize=8)

    for _idx in range(_n_features, len(_axes)):
        _axes[_idx].axis("off")

    plt.suptitle(
        f"Overlap distribusi fitur: {target_class_selector.value} vs Normal "
        f"(semakin menumpuk = semakin sulit dipisahkan fitur ini)"
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(CLASS_NAMES, hybrid_report, mo, target_class_selector):
    if hybrid_report is None:
        _ = mo.md("(confusion matrix belum tersedia untuk analisis ini)")
    else:
        _target_idx = CLASS_NAMES.index(target_class_selector.value)
        _cm_row = hybrid_report["confusion_matrix"][_target_idx]
        _total = sum(_cm_row)

        _lines = [
            f"**Ke mana frame `{target_class_selector.value}` yang sebenarnya "
            f"salah diklasifikasikan (model saat ini):**",
            "",
        ]
        for _cname, _count in zip(CLASS_NAMES, _cm_row):
            _pct = (_count / _total * 100) if _total > 0 else 0.0
            _marker = " <- benar" if _cname == target_class_selector.value else ""
            _lines.append(
                f"- Diprediksi **{_cname}**: {_count:,} ({_pct:.2f}%){_marker}"
            )

        _ = mo.md("\n".join(_lines))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Generalisasi ke Dataset / Kelas Attack Lain

    Notebook ini dirancang supaya bagian analisisnya (distribusi fitur,
    separability ANOVA, embedding scatter, confusion matrix, analisis
    kegagalan per-kelas) **tidak perlu ditulis ulang** kalau nanti:

    - Model gagal di kelas attack yang BEDA (bukan Replay) -- tinggal
      ganti `target_class_selector` di bagian 5.
    - Dataset ganti total (bukan Car Hacking Challenge lagi) -- tinggal
      ubah cell **Config** (`CLASS_NAMES`, `TEMPORAL_FEATURE_COLUMNS`,
      path parquet, `SPATIAL_MODEL_CONFIG`). Selama output pipeline
      data & eval-nya mengikuti kontrak yang sama (kolom `Class`, JSON
      `classification_report` dari `sklearn.metrics.classification_report`,
      checkpoint `SpatialTransformer` dengan `return_embedding=True`),
      seluruh notebook ini langsung jalan tanpa modifikasi lain.

    Kalau nanti pipeline benar-benar pindah ke arsitektur/dataset lain
    yang strukturnya beda total (bukan cuma ganti nama kelas), inilah
    titik alami untuk mulai memisahkan notebook ini jadi versi
    "agnostic" -- sejalan dengan diskusi struktur folder
    `adapters/<dataset>/` dan `model/<method>/` sebelumnya.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Panduan Interpretasi -- Checklist Sebelum Memutuskan Langkah

    | Yang diamati di notebook ini | Kemungkinan penyebab | Langkah yang lebih tepat |
    |---|---|---|
    | Separability ANOVA (bagian 2) RENDAH di semua fitur untuk kelas yang gagal | Fitur belum menangkap pola serangan itu sama sekali | Tambah fitur baru (feature engineering), bukan tuning model |
    | Embedding scatter (bagian 3) kelas gagal BERTUMPUK TOTAL dengan Normal, di PCA **dan** t-SNE | Representasi spatial memang tidak cukup untuk kelas ini | Pertimbangkan naikkan `d_model`/kapasitas, atau fitur tambahan -- BUKAN cuma tuning threshold |
    | Embedding scatter menunjukkan cluster CUKUP terpisah, tapi confusion matrix (bagian 4) masih salah klasifikasi besar-besaran | Masalah di training, bukan representasi | Tuning `class_weight`, threshold, atau monitor metric -- ini yang paling murah dicoba dulu |
    | Histogram overlap (bagian 5) untuk kelas target vs Normal menumpuk penuh di SEMUA fitur | Kelas ini secara fundamental butuh sinyal yang belum ada di fitur manapun sekarang | Riset fitur baru yang menangkap mekanisme spesifik serangan itu (mis. exact-repeat detection untuk Replay) |
    | Precision rendah tapi recall tinggi di banyak kelas attack sekaligus (bagian 4) | Model terlalu agresif memprediksi attack (class_weight kelewat ekstrem) | Lunakkan `class_weight`, atau kalibrasi threshold pasca-training |

    Jalankan notebook ini ulang setiap kali ada perubahan feature
    engineering/model, supaya keputusan berikutnya selalu berdasar bukti
    visual & numerik terbaru -- bukan asumsi dari histori terdahulu.
    """)
    return


if __name__ == "__main__":
    app.run()
