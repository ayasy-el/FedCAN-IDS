"""Single shared ingestion stage with dataset-format adapters."""

import importlib
import fnmatch
from pathlib import Path

from utils.params import load_params


ADAPTERS = {
    "car_hacking_attack_defense": "data.adapters.car_hacking_attack_defense",
    "hcrl_carhacking": "data.adapters.hcrl_carhacking",
    "road_raw": "data.adapters.road_raw",
    "can_mirgu": "data.adapters.can_mirgu",
    "ciciov2024": "data.adapters.ciciov2024",
    "survival_analysis": "data.adapters.survival_analysis",
}
DEFAULT_PATTERNS = {
    "car_hacking_attack_defense": "*.csv",
    "hcrl_carhacking": ("*.csv", "*.txt"),
    "road_raw": "*.log",
    "can_mirgu": "*.log",
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


def _is_excluded(path, raw_dir, patterns):
    """Match exclusions against filename, relative path, or path pattern."""
    relative = path.relative_to(raw_dir).as_posix()
    return any(
        fnmatch.fnmatch(path.name, pattern)
        or fnmatch.fnmatch(relative, pattern)
        or path.match(pattern)
        for pattern in patterns
    )


def run(params):
    dataset = params["dataset"]
    raw_dir = Path(dataset["raw_dir"])
    adapter_name = dataset.get("ingest_adapter", "car_hacking_attack_defense")
    adapter = _load_adapter(adapter_name)
    excluded_files = dataset.get("exclude_files") or []
    if not isinstance(excluded_files, list):
        raise ValueError("dataset.exclude_files must be a list of file patterns")
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
            f"No raw files matching {patterns!r} "
            f"found under {raw_dir}"
        )
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Configured raw files do not exist: {missing}")

    if excluded_files:
        excluded = []
        included = []
        for path in files:
            (excluded if _is_excluded(path, raw_dir, excluded_files) else included).append(path)
        files = included
        if excluded:
            print(
                f"Excluded {len(excluded)} raw files using dataset.exclude_files: "
                f"{excluded_files}"
            )
        if not files:
            raise ValueError("dataset.exclude_files excluded every raw file")

    selector = getattr(adapter, "select_files", None)
    if selector is not None:
        files = selector(files, dataset.get("variant"))
        if not files:
            raise ValueError(
                f"No files remain for {adapter_name!r} variant={dataset.get('variant')!r}"
            )

    output = Path(dataset["interim_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    streaming_writer = getattr(adapter, "write", None)
    if streaming_writer is not None:
        row_count = streaming_writer(files, output)
        print(f"Ingested {len(files)} raw files and {row_count:,} frames")
        return

    result = adapter.read(files).with_row_index("row_id")
    result.write_parquet(output, compression="zstd")
    print(f"Ingested {len(files)} raw files and {len(result):,} frames")


if __name__ == "__main__":
    run(load_params())
