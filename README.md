# FedCAN-IDS

<p align="center">
  <strong>Streaming Causal Intrusion Detection for Controller Area Network traffic</strong>
</p>

<p align="center">
  <a href="https://github.com/ayasy-el/FedCAN-IDS"><img src="https://img.shields.io/badge/project-FedCAN--IDS-1f6feb?style=flat-square" alt="Project"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow 2.21">
  <img src="https://img.shields.io/badge/DVC-reproducible%20pipeline-945DD6?style=flat-square" alt="DVC">
  <img src="https://img.shields.io/badge/MLflow-experiment%20tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow">
</p>

FedCAN-IDS is a machine-learning pipeline for detecting attacks in automotive
CAN bus traffic. It uses a lightweight causal model that classifies one frame
at a time with compressed global and same-ID KV memory.

The project is built around reproducible data processing with DVC, configurable
experiments through `params.yaml`, and MLflow tracking through DagsHub.

> **Project status:** Active research prototype. Results depend on the raw
> dataset, configuration, hardware, and training run.

## Contents

- [Key features](#key-features)
- [Model architecture](#model-architecture)
- [Supported classes](#supported-classes)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Dataset preparation](#dataset-preparation)
- [Quick start](#quick-start)
- [Pipeline stages](#pipeline-stages)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Evaluation and data audits](#evaluation-and-data-audits)
- [Explainability notebook](#explainability-notebook)
- [Experiment tracking](#experiment-tracking)
- [Reproducibility notes](#reproducibility-notes)
- [Contributing](#contributing)
- [License](#license)

## Key features

- Single-frame causal inference with global (64) and same-ID (16) KV memory.
- 11-bit CAN-ID representation projected with Dense(8), plus 11 numerical
  frame/timing features projected with Dense(24).
- Session-aware processing to prevent cross-capture temporal leakage.
- Chronological train/validation/test splitting with row-level leakage checks.
- Train-only normalization statistics reused for validation and test data.
- Class-weighted hybrid training and macro-F1 checkpoint selection for imbalanced
  attack detection.
- DVC pipeline stages for repeatable data preparation, training, and evaluation.
- MLflow metrics and artifacts logged to a DagsHub tracking server.
- Automated reports containing classification metrics and confusion matrices.

## Model architecture

```text
Raw CAN CSV files
        │
        ▼
Ingest and label encoding
        │
        ▼
Session-aware chronological split
        │
        ▼
Temporal feature engineering
        │
        ├──────────────────────────────┐
        ▼                              ▼
Spatial Transformer              Sequence construction
        │                              │
        │                              ▼
        │                         Temporal features
        │                              │
        └──────────────┬───────────────┘
                       ▼
                 Temporal GRU
                       │
                       ▼
                   Classifier
```

Training uses chronological chunks for truncated backpropagation through time;
each frame in a chunk is still processed sequentially and causally.

Temporal normalization statistics are computed from the training split only
and stored in `checkpoints/temporal_norm_stats.json`. Validation and test data
reuse those statistics to avoid data leakage.

## Supported classes

| Label | Class |
|---:|---|
| 0 | Normal |
| 1 | Flooding |
| 2 | Fuzzing |
| 3 | Spoofing |
| 4 | Replay |

## Repository structure

```text
FedCAN-IDS/
├── data/                 # Raw and generated datasets; not committed
├── checkpoints/          # Model checkpoints and MLflow run metadata
├── notebook/             # Marimo explainability application
├── reports/              # JSON metrics and confusion-matrix figures
├── scripts/              # Raw-data and feature-data audit utilities
├── src/
│   ├── data/             # Ingestion, splitting, features, dataset loaders
│   ├── model/            # Causal streaming model and KV memory
│   ├── train_*.py        # Training entry points
│   └── eval_*.py         # Evaluation entry points
├── tests/                # Data integrity and leakage tests
├── dvc.yaml              # Reproducible pipeline definition
├── dvc.lock              # Locked DVC stage state
├── params.yaml           # Dataset, model, training, and MLflow settings
├── requirements.txt      # Runtime and pipeline dependencies
└── README.md
```

## Installation

FedCAN-IDS requires Python compatible with TensorFlow 2.21. A virtual
environment is recommended:

```bash
git clone https://github.com/ayasy-el/FedCAN-IDS.git
cd FedCAN-IDS

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The core dependencies include TensorFlow, Polars, NumPy, scikit-learn,
Matplotlib, Seaborn, PyYAML, DVC, MLflow, and DagsHub.

## Dataset preparation

The pipeline uses the [Car Hacking: Attack & Defense Challenge 2020
Dataset](https://ieee-dataport.org/open-access/car-hacking-attack-defense-challenge-2020-dataset).
Download the dataset from IEEE DataPort, then place the extracted CSV files at:

```text
data/raw/Car_Hacking_Challenge_Dataset/
```

Place the raw CSV files in that directory before running DVC. The raw dataset
is intentionally not included in this repository.

Each source file is treated as a separate `session_id`. This is important
because timestamps are not guaranteed to be globally synchronized between
captures. All order-sensitive operations are partitioned by session.

You can inspect the raw files before ingestion:

```bash
python scripts/audit_raw_data.py
```

## Quick start

From the repository root, run the complete pipeline:

```bash
dvc repro
```

This command executes data preparation, model training, and evaluation using
the dependency graph in `dvc.yaml`.

For manual execution:

```bash
# 1. Prepare the data
python src/data/ingest.py
PYTHONPATH=src python src/data/split.py
PYTHONPATH=src python src/data/feature.py

# 2. Train and evaluate the streaming model
PYTHONPATH=src python src/train_streaming.py
PYTHONPATH=src python src/eval_streaming.py
```


## Pipeline stages

| Stage | Entry point | Description |
|---|---|---|
| Ingest | `src/data/ingest.py` | Reads CSV files, normalizes labels, encodes payload bytes, and writes an interim Parquet dataset. |
| Split | `src/data/split.py` | Creates chronological train, validation, and test splits without duplicated `row_id` values. |
| Feature | `src/data/feature.py` | Adds per-ID and 20 ms window-level temporal features. |
| Streaming training | `src/train_streaming.py` | Trains the causal compressed-KV model. |
| Streaming evaluation | `src/eval_streaming.py` | Evaluates frame-level streaming predictions. |

## Outputs

| Path | Description |
|---|---|
| `data/interim/car_hacking_with_session.parquet` | Ingested dataset with session IDs and encoded labels. |
| `data/processed/car_hacking/{train,val,test}.parquet` | Raw-frame dataset splits. |
| `data/processed/car_hacking/featured/{train,val,test}.parquet` | Splits with engineered temporal features. |
| `checkpoints/streaming_best.keras` | Best streaming checkpoint selected by validation macro-F1. |
| `checkpoints/streaming_final.keras` | Final streaming model. |
| `checkpoints/streaming_norm_stats.json` | Train-derived feature normalization statistics. |
| `reports/metrics/*_classification_report.json` | Accuracy, macro precision, macro recall, macro-F1, and per-class metrics. |
| `reports/figures/*confusion_matrix*.png` | Raw and normalized confusion matrices. |

## Configuration

All primary settings are defined in [`params.yaml`](params.yaml):

- dataset paths;
- validation and test session configuration;
- temporal feature window size;
- Spatial Transformer and hybrid model dimensions;
- sequence length and GRU size;
- batch sizes, learning rates, and epoch counts;
- MLflow tracking URI and experiment names.

After changing parameters, run:

```bash
dvc repro
```

DVC will use the dependency graph and parameter changes to determine which
stages need to be reproduced.

## Evaluation and data audits

Run the feature-data audit after feature engineering:

```bash
python scripts/audit_featured_data.py
```

Run the test suite after the relevant pipeline outputs exist:

```bash
pytest
```

The tests check label completeness, session ID integrity, row-level leakage
between splits, and temporal features at session boundaries. Tests that require
generated data skip cleanly when their input Parquet files are not available.

## Explainability notebook

`notebook/model_explainability.py` is a Marimo application for exploring:

- class distributions across train, validation, and test splits;
- temporal feature distributions by class;
- Spatial Transformer embeddings using PCA or t-SNE;
- confusion matrices and class-specific failure patterns.

Install the notebook-specific packages if needed, then launch it from the
repository root:

```bash
pip install marimo scipy plotly
marimo edit notebook/model_explainability.py
```

The notebook expects generated featured Parquet files, the spatial checkpoint,
and evaluation reports under `reports/`.

## Experiment tracking

Training and evaluation scripts are configured to use MLflow with DagsHub.
Tracking settings are stored under the `mlflow` section of `params.yaml`.

The scripts log model parameters, training metrics, evaluation metrics,
classification reports, confusion matrices, and model artifacts. The training
run ID is stored in `checkpoints/` so evaluation can continue the same MLflow
run when the checkpoint metadata is available.

## Reproducibility notes

- Keep the raw dataset layout unchanged when reproducing an existing run.
- Do not compute normalization statistics independently on validation or test
  data.
- Preserve the configured session-aware ordering; temporal windows must not
  cross session boundaries.
- Keep the model configuration in `params.yaml` synchronized with checkpoints.
- Use `dvc repro` after changing code, data, or tracked parameters.

## Contributing

Contributions are welcome. Before opening a pull request:

1. Keep changes focused and update documentation when behavior changes.
2. Run `git diff --check`.
3. Run `pytest` when generated data is available.
4. Run the relevant DVC stages or `dvc repro` for pipeline changes.
5. Include a concise explanation of any changes to preprocessing, model
   architecture, metrics, or data-splitting behavior.

## License

This project is licensed under the [MIT License](LICENSE).
