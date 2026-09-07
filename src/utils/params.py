from pathlib import Path

import yaml

_PARAMS_CACHE = None


def _find_project_root() -> Path:
    current = Path.cwd()

    for candidate in [current, *current.parents]:
        if (candidate / "params.yaml").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate `params.yaml` in the current directory or any of its "
        "parent directories. Ensure the script is run from within the project "
        "repository containing `params.yaml` and `dvc.yaml`, or invoked via "
        "`dvc repro`."
    )


def load_params(section: str | None = None) -> dict:
    global _PARAMS_CACHE

    if _PARAMS_CACHE is None:
        project_root = _find_project_root()
        params_path = project_root / "params.yaml"

        with open(params_path) as f:
            _PARAMS_CACHE = yaml.safe_load(f)

    if section is None:
        return _PARAMS_CACHE

    value = _PARAMS_CACHE
    for key in section.split("."):
        if key not in value:
            raise KeyError(
                f"Section '{section}' not found in `params.yaml` "
                f"(missing key: '{key}')."
            )
        value = value[key]

    return value
