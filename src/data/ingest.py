"""Single shared ingestion stage with dataset-format adapters."""

import importlib
from pathlib import Path

from utils.params import load_params


ADAPTERS = {
    "car_hacking_attack_defense": "data.adapters.car_hacking_attack_defense",
    "hcrl_carhacking": "data.adapters.hcrl_carhacking",
    "ciciov2024": "data.adapters.ciciov2024",
    "survival_analysis": "data.adapters.survival_analysis",
}
DEFAULT_PATTERNS = {
    "car_hacking_attack_defense": "*.csv",
    "hcrl_carhacking": ("*.csv", "*.txt"),
    "ciciov2024": "*.csv",
    "survival_analysis": "*.txt",
}


def _load_adapter(name):
    try:
        module_name = ADAPTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset.ingest_adapter {name!r}; "
            f"choose one of {sorted(ADAPTERS)}"
        ) from exc
    return importlib.import_module(module_name)


def run(params):
    dataset = params["dataset"]
    raw_dir = Path(dataset["raw_dir"])
    adapter_name = dataset.get("ingest_adapter", "car_hacking_attack_defense")
    adapter = _load_adapter(adapter_name)
    configured_files = dataset.get("raw_files")
    pattern = None
    if configured_files:
        files = []
        for name in configured_files:
            direct = raw_dir / name
            matches = [direct] if direct.exists() else sorted(raw_dir.rglob(name))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one raw file named {name!r} under {raw_dir}, "
                    f"found {len(matches)}"
                )
            files.append(matches[0])
    else:
        patterns = DEFAULT_PATTERNS[adapter_name]
        if isinstance(patterns, str):
            patterns = (patterns,)
        files = sorted(
            {
                path
                for pattern in patterns
                for path in raw_dir.rglob(pattern)
                if not path.name.startswith(".")
            }
        )
    if not files:
        raise FileNotFoundError(
            f"No raw files matching {pattern or 'configured file list'!r} "
            f"found under {raw_dir}"
        )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Configured raw files do not exist: {missing}")

    result = adapter.read(files).with_row_index("row_id")
    output = Path(dataset["interim_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output, compression="zstd")
    print(f"Ingested {len(files)} raw files and {len(result):,} frames")


if __name__ == "__main__":
    run(load_params())
