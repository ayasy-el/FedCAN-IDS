import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    import numpy as np
    import polars as pl

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # Attack Analysis
    """)
    return


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = NOTEBOOK_DIR.parent

    FEATURED_SPLIT_PATHS = {
        "train": PROJECT_ROOT / "data/processed/car_hacking/featured/train.parquet",
        "val": PROJECT_ROOT / "data/processed/car_hacking/featured/val.parquet",
        "test": PROJECT_ROOT / "data/processed/car_hacking/featured/test.parquet",
    }

    CLASS_NAMES = [
        "Normal",
        "Flooding",
        "Fuzzing",
        "Spoofing",
        "Replay",
    ]

    BYTE_COLUMNS = [f"Data_{i}" for i in range(8)]

    # Kontras tinggi dan konsisten di seluruh plot.
    CLASS_COLORS = {
        "Normal": "#1976D2",
        "Flooding": "#D32F2F",
        "Fuzzing": "#2E7D32",
        "Spoofing": "#FFEA00",
        "Replay": "#F57C00",
    }
    return BYTE_COLUMNS, CLASS_COLORS, CLASS_NAMES, FEATURED_SPLIT_PATHS


@app.cell
def _(mo):
    split_selector = mo.ui.dropdown(
        options=[
            "train",
            "val",
            "test",
        ],
        value="test",
        label="Split",
    )

    split_selector
    return (split_selector,)


@app.cell
def _(FEATURED_SPLIT_PATHS, pl, split_selector):
    df_raw = pl.read_parquet(FEATURED_SPLIT_PATHS[split_selector.value])
    return (df_raw,)


@app.cell
def _(BYTE_COLUMNS, df_raw, pl):
    _byte_exprs = [
        pl.col(c)
        .map_elements(
            lambda x: None if x == "PAD" else int(x, 16),
            return_dtype=pl.Int32,
        )
        .alias(f"{c}_val")
        for c in BYTE_COLUMNS
    ]

    df = df_raw.with_columns(
        pl.col("Arbitration_ID")
        .map_elements(
            lambda x: int(x, 16),
            return_dtype=pl.Int64,
        )
        .alias("Arbitration_ID_int"),
        *_byte_exprs,
    )
    return (df,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Frekuensi rata-rata per ID
    """)
    return


@app.cell
def _(mo):
    benign_only_toggle = mo.ui.checkbox(
        value=True,
        label="Benign only",
    )

    benign_only_toggle
    return (benign_only_toggle,)


@app.cell
def _(benign_only_toggle, df, go, mo, np, pl):
    # ========================================================
    # FILTER
    # ========================================================

    if benign_only_toggle.value:
        _d = df.filter(
            pl.col("Class") == 0
        )
    else:
        _d = df

    # ========================================================
    # STATISTICS
    # ========================================================

    _stats = (
        _d
        .filter(
            pl.col("Delta_Id") > 0
        )
        .group_by("Arbitration_ID")
        .agg(
            pl.col("Delta_Id")
            .mean()
            .alias("mean_Deltatime")
        )
        .with_columns(
            (
                1000.0
                / pl.col("mean_Deltatime")
            ).alias("mean_freq_hz")
        )
        .sort("mean_freq_hz")
    )

    _labels = [
        "0x" + x
        for x in _stats[
            "Arbitration_ID"
        ].to_list()
    ]

    _y = (
        _stats[
            "mean_freq_hz"
        ]
        .to_numpy()
    )

    _x = np.arange(
        len(_y)
    )

    # ========================================================
    # ATTACK-ONLY ARBITRATION IDS
    # ========================================================
    #
    # ID yang muncul pada attack tetapi tidak pernah muncul
    # pada traffic benign.
    # ========================================================

    _benign_ids = set(
        df
        .filter(
            pl.col("Class") == 0
        )
        ["Arbitration_ID"]
        .unique()
        .to_list()
    )

    _attack_ids = set(
        df
        .filter(
            pl.col("Class") != 0
        )
        ["Arbitration_ID"]
        .unique()
        .to_list()
    )

    _attack_only_ids = sorted(
        _attack_ids - _benign_ids,
        key=lambda x: int(x, 16),
    )

    _attack_only_labels = [
        "0x" + x
        for x in _attack_only_ids
    ]

    # ========================================================
    # INFO
    # ========================================================

    if _attack_only_labels:

        _attack_only_text = ", ".join(
            f"`{x}`"
            for x in _attack_only_labels
        )

    else:

        _attack_only_text = "`None`"

    _info = mo.md(
        f"""
        **Total Arbitration ID:** `{len(_stats):,}`

        **New Arbitration ID in Attack:**
        {_attack_only_text}
        """
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig_frequency = go.Figure()

    # --------------------------------------------------------
    # Stem
    # --------------------------------------------------------

    for _xi, _yi in zip(
        _x,
        _y,
    ):
        fig_frequency.add_trace(
            go.Scatter(
                x=[
                    _xi,
                    _xi,
                ],
                y=[
                    1e-9,
                    _yi,
                ],
                mode="lines",
                line=dict(
                    width=1,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    fig_frequency.add_trace(
        go.Scatter(
            x=_x,
            y=_y,
            mode="markers",
            marker=dict(
                size=6,
            ),
            text=_labels,
            hovertemplate=(
                "ID=%{text}"
                "<br>Frequency=%{y:.3f} Hz"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # ========================================================
    # AXES
    # ========================================================

    fig_frequency.update_yaxes(
        type="log",
        title="Mean frequency (Hz)",
    )

    fig_frequency.update_xaxes(
        tickmode="array",
        tickvals=_x,
        ticktext=_labels,
        tickangle=-90,
        title="Arbitration ID",
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig_frequency.update_layout(
        title=(
            "Mean frequency per Arbitration ID"
            + (
                " — Benign only"
                if benign_only_toggle.value
                else " — All traffic"
            )
        ),
        template="plotly_white",
        height=500,
        margin=dict(
            l=70,
            r=20,
            t=60,
            b=130,
        ),
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    mo.vstack(
        [
            _info,
            fig_frequency,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Inter-frame arrival time
    """)
    return


@app.cell
def _(df, mo, pl):
    _id_counts = (
        df.group_by("Arbitration_ID")
        .agg(pl.len().alias("n"))
        .sort(
            "n",
            descending=True,
        )
    )

    _id_options = ["0x" + s for s in _id_counts["Arbitration_ID"].to_list()]
    _id_options = sorted(_id_options, key=lambda x: int(x[2:], 16))

    id_selector = mo.ui.dropdown(
        options=_id_options,
        searchable=True,
        value=(_id_options[0] if _id_options else None),
        label="Arbitration ID",
    )

    _present_classes = sorted(df["Class"].unique().to_list())

    _attack_options = ["All"] + [c for c in _present_classes if c != 0]

    attack_class_selector = mo.ui.dropdown(
        options=_attack_options,
        value="All",
        label="Attack class",
    )

    mo.hstack(
        [
            id_selector,
            attack_class_selector,
        ]
    )
    return attack_class_selector, id_selector


@app.cell
def _(np):
    def swarm_positions(
        values,
        max_width=0.42,
        n_bins=100,
    ):
        values = np.asarray(
            values,
            dtype=float,
        )

        n = len(values)

        if n == 0:
            return np.array([])

        if n == 1:
            return np.array([0.0])

        _vmin = np.nanmin(values)

        _vmax = np.nanmax(values)

        if _vmax == _vmin:
            _bin_ids = np.zeros(
                n,
                dtype=int,
            )

        else:
            _bin_ids = np.floor((values - _vmin) / (_vmax - _vmin) * n_bins).astype(int)

            _bin_ids = np.clip(
                _bin_ids,
                0,
                n_bins - 1,
            )

        positions = np.zeros(
            n,
            dtype=float,
        )

        for _bin in np.unique(_bin_ids):
            _idx = np.where(_bin_ids == _bin)[0]

            _count = len(_idx)

            if _count == 1:
                _x = np.array([0.0])

            else:
                _step = min(
                    0.055,
                    max_width
                    / max(
                        1,
                        (_count - 1) / 2,
                    ),
                )

                _x = (np.arange(_count) - (_count - 1) / 2) * _step

                _x = np.clip(
                    _x,
                    -max_width,
                    max_width,
                )

            positions[_idx] = _x

        return positions

    return (swarm_positions,)


@app.cell
def _(
    CLASS_COLORS,
    CLASS_NAMES,
    attack_class_selector,
    df,
    go,
    id_selector,
    np,
    pl,
    swarm_positions,
):
    _target_id = id_selector.value[2:] if id_selector.value else None

    _sub = df.filter(pl.col("Arbitration_ID") == _target_id)

    _MAX_POINTS = 4000

    _rng = np.random.default_rng(42)

    _normal = _sub.filter(pl.col("Class") == 0)["Delta_Id"].to_numpy() * 1000

    if len(_normal) > _MAX_POINTS:
        _normal = _rng.choice(
            _normal,
            _MAX_POINTS,
            replace=False,
        )

    _normal_x = swarm_positions(_normal)

    fig_swarm = go.Figure()

    fig_swarm.add_trace(
        go.Scattergl(
            x=_normal_x,
            y=_normal,
            mode="markers",
            name="Normal",
            marker=dict(
                size=5,
                opacity=0.55,
                color=CLASS_COLORS["Normal"],
            ),
            hovertemplate=("Normal<br>time=%{y:.3f} ms<extra></extra>"),
        )
    )

    if attack_class_selector.value == "All":
        _attack_classes = [
            c for c in sorted(_sub["Class"].unique().to_list()) if c != 0
        ]

    else:
        _attack_classes = [attack_class_selector.value]

    for _cls_idx in _attack_classes:
        _attack = _sub.filter(pl.col("Class") == _cls_idx)["Delta_Id"].to_numpy() * 1000

        if len(_attack) == 0:
            continue

        if len(_attack) > _MAX_POINTS:
            _attack = _rng.choice(
                _attack,
                _MAX_POINTS,
                replace=False,
            )

        _attack_x = swarm_positions(_attack)

        _attack_name = CLASS_NAMES[_cls_idx]

        fig_swarm.add_trace(
            go.Scattergl(
                x=_attack_x,
                y=_attack,
                mode="markers",
                name=_attack_name,
                marker=dict(
                    size=5,
                    opacity=0.7,
                    color=CLASS_COLORS.get(
                        _attack_name,
                        "#888888",
                    ),
                ),
                hovertemplate=(f"{_attack_name}<br>time=%{{y:.3f}} ms<extra></extra>"),
            )
        )

    _attack_label = (
        "All attack classes"
        if (attack_class_selector.value == "All")
        else CLASS_NAMES[attack_class_selector.value]
    )

    fig_swarm.update_xaxes(
        range=[
            -0.5,
            0.5,
        ],
        tickmode="array",
        tickvals=[0],
        ticktext=[id_selector.value],
        title="Arbitration ID",
    )

    fig_swarm.update_yaxes(title=("Inter-frame arrival time (ms)"))

    fig_swarm.update_layout(
        title=(f"Inter-frame arrival time — {id_selector.value} — {_attack_label}"),
        template="plotly_white",
        height=550,
        width=650,
        legend=dict(
            orientation="h",
            y=1.02,
            x=0,
        ),
        margin=dict(
            l=70,
            r=20,
            t=70,
            b=70,
        ),
    )

    fig_swarm
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Time Series Nilai Byte
    """)
    return


@app.cell
def _(df, mo):
    _sessions = df["session_id"].unique().sort().to_list()

    session_selector = mo.ui.dropdown(
        options=_sessions,
        value=(_sessions[0] if _sessions else None),
        label="Session",
    )

    session_selector
    return (session_selector,)


@app.cell
def _(df, mo, pl, session_selector):
    _session_df = df.filter(pl.col("session_id") == session_selector.value)

    if len(_session_df) > 0:
        _t_min = float(_session_df["Timestamp"].min())

        _t_max = float(_session_df["Timestamp"].max())

    else:
        _t_min = 0.0
        _t_max = 1.0

    _range = max(
        0.001,
        _t_max - _t_min,
    )

    _default_duration = min(
        0.3,
        _range,
    )

    time_window_start = mo.ui.slider(
        start=_t_min,
        stop=_t_max,
        value=_t_min,
        step=max(
            _range / 1000,
            0.000001,
        ),
        label="Waktu mulai",
        show_value=True,
    )

    time_window_duration = mo.ui.number(
        start=0.001,
        stop=_range,
        value=_default_duration,
        step=max(
            _range / 1000,
            0.001,
        ),
        label="Durasi (s)",
    )

    mo.vstack(
        [
            time_window_start,
            time_window_duration,
        ]
    )
    return time_window_duration, time_window_start


@app.cell
def _(
    CLASS_COLORS,
    CLASS_NAMES,
    df,
    go,
    id_selector,
    make_subplots,
    np,
    pl,
    session_selector,
    time_window_duration,
    time_window_start,
):
    _target_id = id_selector.value[2:] if id_selector.value else None

    _t0 = time_window_start.value

    _t1 = _t0 + time_window_duration.value

    _win = df.filter(
        (pl.col("session_id") == session_selector.value)
        & (pl.col("Arbitration_ID") == _target_id)
        & (pl.col("Timestamp") >= _t0)
        & (pl.col("Timestamp") <= _t1)
    ).sort("Timestamp")

    if len(_win) == 0:
        fig_byte_timeseries = go.Figure()

        fig_byte_timeseries.add_annotation(
            text="No data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    else:
        _dlc = int(_win["DLC"][0])

        _active_bytes = [f"Data_{i}_val" for i in range(_dlc)]

        _fig = make_subplots(
            rows=len(_active_bytes),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f"Byte{i}" for i in range(_dlc)],
        )

        _t = _win["Timestamp"].to_numpy()

        _classes = _win["Class"].to_numpy()

        _MAX_POINTS = 3000

        for _i, _col in enumerate(_active_bytes):
            _row = _i + 1

            _vals = _win[_col].to_numpy()

            for _cls_idx in sorted(set(_classes.tolist())):
                _indices = np.where(_classes == _cls_idx)[0]

                if len(_indices) > _MAX_POINTS:
                    _sample = np.linspace(
                        0,
                        len(_indices) - 1,
                        _MAX_POINTS,
                    ).astype(int)

                    _indices = _indices[_sample]

                _cname = CLASS_NAMES[_cls_idx]

                _fig.add_trace(
                    go.Scattergl(
                        x=_t[_indices],
                        y=_vals[_indices],
                        mode="lines+markers",
                        name=_cname,
                        legendgroup=_cname,
                        showlegend=(_row == 1),
                        line=dict(
                            width=1.2,
                            color=CLASS_COLORS.get(
                                _cname,
                                "#888888",
                            ),
                        ),
                        marker=dict(
                            size=4,
                        ),
                        hovertemplate=(
                            f"{_cname}<br>t=%{{x:.6f}}s<br>value=%{{y}}<extra></extra>"
                        ),
                    ),
                    row=_row,
                    col=1,
                )

            _fig.update_yaxes(
                range=[
                    0,
                    255,
                ],
                title_text=f"B{_i}",
                row=_row,
                col=1,
            )

        _fig.update_xaxes(
            title_text="Time (s)",
            row=len(_active_bytes),
            col=1,
        )

        _fig.update_layout(
            title=(f"{id_selector.value} — {session_selector.value}"),
            template="plotly_white",
            height=max(
                350,
                150 * len(_active_bytes),
            ),
            hovermode="closest",
            margin=dict(
                l=60,
                r=20,
                t=70,
                b=50,
            ),
        )

        fig_byte_timeseries = _fig

    fig_byte_timeseries
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Raster Scatter Byte vs Waktu
    """)
    return


@app.cell
def _(
    CLASS_COLORS,
    CLASS_NAMES,
    df,
    go,
    id_selector,
    make_subplots,
    np,
    pl,
    session_selector,
):
    _target_id = id_selector.value[2:] if id_selector.value else None

    _sub = df.filter(
        (pl.col("session_id") == session_selector.value)
        & (pl.col("Arbitration_ID") == _target_id)
    ).sort("Timestamp")

    if len(_sub) == 0:
        fig_raster = go.Figure()

        fig_raster.add_annotation(
            text="No data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    else:
        _dlc = int(_sub["DLC"][0])

        _active_bytes = [f"Data_{i}_val" for i in range(_dlc)]

        _t0 = _sub["Timestamp"].min()

        _elapsed_ms = (_sub["Timestamp"] - _t0).to_numpy() * 1000

        _classes = _sub["Class"].to_numpy()

        _n_cols = 4

        _n_rows = (len(_active_bytes) + _n_cols - 1) // _n_cols

        _fig = make_subplots(
            rows=_n_rows,
            cols=_n_cols,
            subplot_titles=[f"Byte{i}" for i in range(len(_active_bytes))],
            horizontal_spacing=0.06,
            vertical_spacing=0.12,
        )

        _MAX_POINTS = 1500

        for _i, _col in enumerate(_active_bytes):
            _row = (_i // _n_cols) + 1

            _col_idx = (_i % _n_cols) + 1

            _vals = _sub[_col].to_numpy()

            for _cls_idx in sorted(set(_classes.tolist())):
                _indices = np.where(_classes == _cls_idx)[0]

                if len(_indices) > _MAX_POINTS:
                    _sample = np.linspace(
                        0,
                        len(_indices) - 1,
                        _MAX_POINTS,
                    ).astype(int)

                    _indices = _indices[_sample]

                _cname = CLASS_NAMES[_cls_idx]

                _fig.add_trace(
                    go.Scattergl(
                        x=_elapsed_ms[_indices],
                        y=_vals[_indices],
                        mode="markers",
                        name=_cname,
                        legendgroup=_cname,
                        showlegend=(_i == 0),
                        marker=dict(
                            size=4,
                            opacity=0.6,
                            color=CLASS_COLORS.get(
                                _cname,
                                "#888888",
                            ),
                        ),
                        hovertemplate=(
                            f"{_cname}"
                            "<br>"
                            f"Byte B{_i}"
                            "<br>"
                            "t=%{x:.2f} ms"
                            "<br>"
                            "value=%{y}"
                            "<extra></extra>"
                        ),
                    ),
                    row=_row,
                    col=_col_idx,
                )

            _fig.update_yaxes(
                range=[
                    0,
                    255,
                ],
                title_text="Value",
                row=_row,
                col=_col_idx,
            )

            _fig.update_xaxes(
                title_text="Time (ms)",
                row=_row,
                col=_col_idx,
            )

        _fig.update_layout(
            title=(f"{id_selector.value} — {session_selector.value}"),
            template="plotly_white",
            height=(240 * _n_rows),
            margin=dict(
                l=50,
                r=20,
                t=70,
                b=60,
            ),
        )

        fig_raster = _fig

    fig_raster
    return


@app.cell
def _(np):
    def classify_byte_pattern(
        values: np.ndarray,
    ) -> str:

        values = values[~np.isnan(values)]

        if len(values) == 0:
            return "No byte"

        n_unique = len(np.unique(values))

        if n_unique == 1:
            return "Constant"

        diffs = np.diff(values)

        if len(diffs) == 0:
            return "Constant"

        frac_increasing = np.mean(diffs > 0)

        frac_decreasing = np.mean(diffs < 0)

        monotonic_frac = max(
            frac_increasing,
            frac_decreasing,
        )

        if monotonic_frac > 0.75 and n_unique > 8:
            return "Sawtooth"

        if n_unique <= 8:
            return "Random"

        return "Polynomial"

    return (classify_byte_pattern,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Byte type per ID
    """)
    return


@app.cell
def _(df, mo):
    # --------------------------------------------------------
    # Ambil seluruh Arbitration ID dan urutkan secara numerik.
    # --------------------------------------------------------

    _sorted_ids = (
        df.select("Arbitration_ID_int")
        .unique()
        .sort("Arbitration_ID_int")["Arbitration_ID_int"]
        .to_list()
    )

    # --------------------------------------------------------
    # Dictionary:
    #
    # None -> tampilkan top N ID
    # ID tertentu -> tampilkan hanya ID tersebut
    #
    # Label yang ditampilkan:
    #   None
    #   0x001
    #   0x100
    #   ...
    # --------------------------------------------------------

    _id_options = {
        "None": None,
        **{f"0x{_id:03X}": _id for _id in _sorted_ids},
    }

    heatmap_id_selector = mo.ui.dropdown(
        options=_id_options,
        value=None,
        label="Arbitration ID",
        searchable=True,
    )

    heatmap_benign_only = mo.ui.checkbox(
        value=True,
        label="Benign only",
    )

    n_ids_for_heatmap = mo.ui.slider(
        start=5,
        stop=56,
        step=1,
        value=20,
        label="Jumlah ID",
        show_value=True,
    )

    mo.hstack(
        [
            heatmap_id_selector,
            heatmap_benign_only,
            n_ids_for_heatmap,
        ]
    )
    return heatmap_benign_only, heatmap_id_selector, n_ids_for_heatmap


@app.cell
def _(
    classify_byte_pattern,
    df,
    go,
    heatmap_benign_only,
    heatmap_id_selector,
    n_ids_for_heatmap,
    np,
    pl,
):
    # ========================================================
    # FILTER TRAFFIC
    # ========================================================

    if heatmap_benign_only.value:
        _heatmap_df = df.filter(pl.col("Class") == 0)

    else:
        _heatmap_df = df

    # ========================================================
    # SELECT IDS
    # ========================================================

    if heatmap_id_selector.value is None:
        # ----------------------------------------------------
        # None:
        # gunakan top N ID berdasarkan jumlah frame.
        # ----------------------------------------------------

        _top_ids = (
            _heatmap_df.group_by("Arbitration_ID")
            .agg(
                pl.len().alias("n"),
                pl.col("DLC").max().alias("dlc"),
            )
            .sort(
                "n",
                descending=True,
            )
            .head(n_ids_for_heatmap.value)
            .sort("Arbitration_ID")
        )

    else:
        # ----------------------------------------------------
        # ID tertentu dipilih.
        # ----------------------------------------------------

        _selected_id_int = heatmap_id_selector.value

        # Konversi integer -> format Arbitration_ID
        # yang digunakan dataframe.
        _selected_id = f"{_selected_id_int:03X}"

        _top_ids = (
            _heatmap_df.filter(pl.col("Arbitration_ID") == _selected_id)
            .group_by("Arbitration_ID")
            .agg(
                pl.len().alias("n"),
                pl.col("DLC").max().alias("dlc"),
            )
        )

    # ========================================================
    # CATEGORY
    # ========================================================

    _category_to_code = {
        "Constant": 0,
        "Sawtooth": 1,
        "Polynomial": 2,
        "Random": 3,
        "No byte": 4,
    }

    _category_names = list(_category_to_code.keys())

    _category_colors = [
        "#42A5F5",  # Constant
        "#66BB6A",  # Sawtooth
        "#FFA726",  # Polynomial
        "#EC407A",  # Random
        "#424242",  # No byte
    ]

    # ========================================================
    # EMPTY RESULT
    # ========================================================

    if len(_top_ids) == 0:
        fig_heatmap = go.Figure()

        fig_heatmap.add_annotation(
            text=("No data for selected Arbitration ID"),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

        fig_heatmap.update_layout(
            template="plotly_white",
            height=400,
        )

    else:
        # ====================================================
        # IDS
        # ====================================================

        _ids = _top_ids["Arbitration_ID"].to_list()

        _dlcs = _top_ids["dlc"].to_list()

        # ====================================================
        # MATRIX
        # ====================================================

        _matrix = np.zeros(
            (
                8,
                len(_ids),
            ),
            dtype=int,
        )

        for _col_idx, (
            _aid,
            _dlc,
        ) in enumerate(
            zip(
                _ids,
                _dlcs,
            )
        ):
            _frames = _heatmap_df.filter(pl.col("Arbitration_ID") == _aid)

            for _byte_idx in range(8):
                # --------------------------------------------
                # Byte tidak digunakan oleh ID.
                # --------------------------------------------

                if _byte_idx >= _dlc:
                    _matrix[
                        7 - _byte_idx,
                        _col_idx,
                    ] = _category_to_code["No byte"]

                    continue

                # --------------------------------------------
                # Ambil nilai byte.
                # --------------------------------------------

                _vals = _frames[f"Data_{_byte_idx}_val"].to_numpy().astype(float)

                _cat = classify_byte_pattern(_vals)

                _matrix[
                    7 - _byte_idx,
                    _col_idx,
                ] = _category_to_code[_cat]

        # ====================================================
        # COLORSCALE
        # ====================================================

        _colorscale = []

        for (
            _code,
            _color,
        ) in enumerate(_category_colors):
            _left = _code / 4

            _right = (_code + 1) / 4 if _code < 4 else 1

            _colorscale.extend(
                [
                    [
                        _left,
                        _color,
                    ],
                    [
                        _right,
                        _color,
                    ],
                ]
            )

        # ====================================================
        # CUSTOMDATA
        # ====================================================

        _customdata = [
            [
                _category_names[
                    _matrix[
                        _row,
                        _col,
                    ]
                ]
                for _col in range(len(_ids))
            ]
            for _row in range(8)
        ]

        # ====================================================
        # FIGURE
        # ====================================================

        fig_heatmap = go.Figure()

        fig_heatmap.add_trace(
            go.Heatmap(
                z=_matrix,
                x=["0x" + _id for _id in _ids],
                y=[
                    f"B{i}"
                    for i in range(
                        7,
                        -1,
                        -1,
                    )
                ],
                zmin=0,
                zmax=4,
                colorscale=_colorscale,
                showscale=False,
                xgap=1,
                ygap=1,
                customdata=_customdata,
                hovertemplate=("ID=%{x}<br>%{y}<br>%{customdata}<extra></extra>"),
            )
        )

        # ====================================================
        # LEGEND
        # ====================================================

        for (
            _code,
            _name,
        ) in enumerate(_category_names):
            fig_heatmap.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=(_category_colors[_code]),
                    ),
                    name=_name,
                )
            )

        # ====================================================
        # AXES
        # ====================================================

        fig_heatmap.update_xaxes(
            tickangle=-90,
            title="Arbitration ID",
        )

        fig_heatmap.update_yaxes(
            title="Byte",
        )

        # ====================================================
        # TITLE
        # ====================================================

        if heatmap_id_selector.value is None:
            _selection_text = f"Top {len(_ids)} IDs"

        else:
            _selection_text = f"ID 0x{heatmap_id_selector.value:03X}"

        fig_heatmap.update_layout(
            title=(
                "Byte type per ID — "
                + ("Benign only" if heatmap_benign_only.value else "All traffic")
                + " — "
                + _selection_text
            ),
            template="plotly_white",
            height=500,
            width=max(
                900,
                len(_ids) * 45,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(
                l=60,
                r=20,
                t=60,
                b=120,
            ),
        )

    fig_heatmap
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Session Attack Overview
    """)
    return


@app.cell
def _(CLASS_NAMES, df, mo, pl, session_selector):
    _session = session_selector.value

    _session_df = df.filter(pl.col("session_id") == _session).sort("Timestamp")

    _present_classes = sorted(_session_df["Class"].unique().to_list())

    _attack_classes = [c for c in _present_classes if c != 0]

    if _attack_classes:
        _attack_text = ", ".join(CLASS_NAMES[c] for c in _attack_classes)
    else:
        _attack_text = "None"

    if len(_session_df):
        _t0 = float(_session_df["Timestamp"].min())

        _t1 = float(_session_df["Timestamp"].max())

        _duration = _t1 - _t0

    else:
        _t0 = 0
        _t1 = 0
        _duration = 0

    mo.md(
        f"""
        **Session:** `{_session}`

        **Attack classes:** `{_attack_text}`

        **Duration:** `{_duration:.3f} s`

        **Frames:** `{len(_session_df):,}`
        """
    )
    return


@app.cell
def _(
    CLASS_COLORS,
    CLASS_NAMES,
    df,
    go,
    make_subplots,
    np,
    pl,
    session_selector,
):
    _session = session_selector.value

    _session_df = df.filter(pl.col("session_id") == _session).sort("Timestamp")

    if len(_session_df) == 0:
        fig_session_overview = go.Figure()

        fig_session_overview.add_annotation(
            text="No data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    else:
        # ====================================================
        # ORIGINAL TIME
        # ====================================================

        _original_t0 = float(_session_df["Timestamp"].min())

        _original_t1 = float(_session_df["Timestamp"].max())

        # ====================================================
        # DOWNSAMPLE
        # ====================================================

        _MAX_POINTS = 10000

        _n = len(_session_df)

        if _n > _MAX_POINTS:
            _sample_idx = np.linspace(
                0,
                _n - 1,
                _MAX_POINTS,
            ).astype(int)

            _session_df = _session_df[_sample_idx]

        # ====================================================
        # TIME
        # ====================================================

        _time_ms = (_session_df["Timestamp"].to_numpy() - _original_t0) * 1000

        _classes = _session_df["Class"].to_numpy()

        # ====================================================
        # ARBITRATION ID
        # ====================================================

        _arbitration_ids = _session_df["Arbitration_ID_int"].to_numpy()

        # ====================================================
        # FIGURE
        #
        # 1. Payload
        # 2. Arbitration ID
        # 3. Attack timeline
        # ====================================================

        fig_session_overview = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[
                0.58,
                0.22,
                0.20,
            ],
            vertical_spacing=0.06,
            subplot_titles=[
                "Payload bytes",
                "Arbitration ID",
                "Attack timeline",
            ],
        )

        # ====================================================
        # 1. PAYLOAD BYTES
        # ====================================================

        for _byte_idx in range(8):
            _values = _session_df[f"Data_{_byte_idx}_val"].to_numpy()

            # B0 -> 0 ... 1
            # B1 -> 1 ... 2
            # ...
            # B7 -> 7 ... 8

            _y = _byte_idx + (_values / 255.0)

            for _class_id in sorted(set(_classes.tolist())):
                _mask = _classes == _class_id

                if not np.any(_mask):
                    continue

                _class_name = CLASS_NAMES[_class_id]

                fig_session_overview.add_trace(
                    go.Scattergl(
                        x=_time_ms[_mask],
                        y=_y[_mask],
                        mode="markers",
                        name=_class_name,
                        legendgroup=_class_name,
                        # Legend hanya muncul sekali.
                        showlegend=(_byte_idx == 0),
                        marker=dict(
                            size=4,
                            opacity=0.65,
                            color=CLASS_COLORS.get(
                                _class_name,
                                "#888888",
                            ),
                        ),
                        customdata=(_values[_mask]),
                        hovertemplate=(
                            f"Class: {_class_name}"
                            f"<br>Byte: B{_byte_idx}"
                            "<br>Value: %{customdata}"
                            "<br>Time: %{x:.2f} ms"
                            "<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

        # ====================================================
        # PAYLOAD Y AXIS
        # ====================================================

        fig_session_overview.update_yaxes(
            row=1,
            col=1,
            tickmode="array",
            tickvals=[i + 0.5 for i in range(8)],
            ticktext=[f"B{i}" for i in range(8)],
            range=[
                0,
                8,
            ],
            title="Byte",
        )

        for _i in range(1, 8):
            fig_session_overview.add_hline(
                y=_i,
                row=1,
                col=1,
                line_width=1,
                line_dash="dot",
                opacity=0.3,
            )

        # ====================================================
        # 2. ARBITRATION ID VS TIME
        # ====================================================
        #
        # X = waktu relatif session
        # Y = Arbitration ID
        # Color = class
        # ====================================================

        for _class_id in sorted(set(_classes.tolist())):
            _mask = _classes == _class_id

            if not np.any(_mask):
                continue

            _class_name = CLASS_NAMES[_class_id]

            _ids = _arbitration_ids[_mask]

            # ------------------------------------------------
            # Hover ID hexadecimal.
            # ------------------------------------------------

            _hover_ids = np.array(
                [f"0x{int(_id):03X}" for _id in _ids],
                dtype=object,
            )

            fig_session_overview.add_trace(
                go.Scattergl(
                    x=_time_ms[_mask],
                    y=_ids,
                    mode="markers",
                    name=_class_name,
                    legendgroup=_class_name,
                    # Legend sudah dibuat oleh payload.
                    showlegend=False,
                    marker=dict(
                        size=4,
                        opacity=0.65,
                        color=CLASS_COLORS.get(
                            _class_name,
                            "#888888",
                        ),
                    ),
                    customdata=_hover_ids,
                    hovertemplate=(
                        f"Class: {_class_name}"
                        "<br>Arbitration ID: "
                        "%{customdata}"
                        "<br>ID decimal: %{y}"
                        "<br>Time: %{x:.2f} ms"
                        "<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

        # ====================================================
        # ARBITRATION ID AXIS
        # ====================================================

        fig_session_overview.update_yaxes(
            row=2,
            col=1,
            title="Arbitration ID",
            rangemode="tozero",
        )

        # ====================================================
        # 3. ATTACK TIMELINE
        # ====================================================
        #
        # Setiap class mempunyai lane sendiri sehingga
        # class yang sparse tetap terlihat.
        #
        # 0 = Normal
        # 1 = Flooding
        # 2 = Fuzzing
        # 3 = Spoofing
        # 4 = Replay
        # ====================================================

        _timeline_classes = [
            0,
            1,
            2,
            3,
            4,
        ]

        for _class_id in _timeline_classes:
            _class_name = CLASS_NAMES[_class_id]

            _class_mask = _classes == _class_id

            if not np.any(_class_mask):
                continue

            # ------------------------------------------------
            # Index frame dari class.
            # ------------------------------------------------

            _class_indices = np.where(_class_mask)[0]

            # ------------------------------------------------
            # Cari segment consecutive.
            # ------------------------------------------------

            _breaks = np.where(np.diff(_class_indices) > 1)[0] + 1

            _segments = np.split(
                _class_indices,
                _breaks,
            )

            _x = []
            _y = []

            for _segment in _segments:
                if len(_segment) == 0:
                    continue

                _start_idx = _segment[0]
                _end_idx = _segment[-1]

                _start_time = _time_ms[_start_idx]

                _end_time = _time_ms[_end_idx]

                # Single frame.
                if _start_time == _end_time:
                    _end_time = _start_time + 0.01

                _x.extend(
                    [
                        _start_time,
                        _end_time,
                        None,
                    ]
                )

                _y.extend(
                    [
                        _class_id,
                        _class_id,
                        None,
                    ]
                )

            # ------------------------------------------------
            # Satu trace untuk satu class.
            # ------------------------------------------------

            fig_session_overview.add_trace(
                go.Scattergl(
                    x=_x,
                    y=_y,
                    mode="lines",
                    name=_class_name,
                    legendgroup=_class_name,
                    showlegend=False,
                    line=dict(
                        width=14,
                        color=CLASS_COLORS.get(
                            _class_name,
                            "#888888",
                        ),
                    ),
                    hovertemplate=(
                        f"Class: {_class_name}<br>Time: %{{x:.2f}} ms<extra></extra>"
                    ),
                    connectgaps=False,
                ),
                row=3,
                col=1,
            )

        # ====================================================
        # TIMELINE Y AXIS
        # ====================================================

        fig_session_overview.update_yaxes(
            row=3,
            col=1,
            tickmode="array",
            tickvals=[
                0,
                1,
                2,
                3,
                4,
            ],
            ticktext=[
                "Normal",
                "Flooding",
                "Fuzzing",
                "Spoofing",
                "Replay",
            ],
            range=[
                -0.6,
                4.6,
            ],
            title="Class",
        )

        # ====================================================
        # X AXIS
        # ====================================================

        fig_session_overview.update_xaxes(
            row=3,
            col=1,
            title="Time from session start (ms)",
        )

        # ====================================================
        # LAYOUT
        # ====================================================

        fig_session_overview.update_layout(
            title=(
                f"Session {_session} — Payload, Arbitration ID, and Attack Timeline"
            ),
            template="plotly_white",
            height=1000,
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(
                l=100,
                r=30,
                t=120,
                b=70,
            ),
        )

    fig_session_overview
    return


if __name__ == "__main__":
    app.run()
