"""
How to use this module:

    from utils.params import load_params

    p = load_params()                    # all contents of params.yaml
    model_cfg = load_params("model.spatial")   # only a specific section (dot notation)
"""

from pathlib import Path

import yaml

_PARAMS_CACHE = None


def _find_project_root() -> Path:
    """
    Find the project root by walking up from the current working directory
    until a directory containing `params.yaml` is found.

    This keeps path resolution consistent regardless of where the loader
    is invoked, whether through `dvc repro` or directly from the project root.
    """
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
    """
    Load `params.yaml` with caching to avoid repeated file reads within
    the same process.

    Parameters
    ----------
    section : str, optional
        Dot-delimited path to load a specific subsection, e.g. `model.streaming`
        or `training.streaming`. If `None`, return the entire contents of the file.
    """
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
