"""Backward-compatible entry point; feature generation now lives in prepare."""

from data.prepare import run
from utils.params import load_params


if __name__ == "__main__":
    run(load_params())
