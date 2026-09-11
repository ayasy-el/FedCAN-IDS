import argparse

from utils.params import load_params
from data.adapters import can_bigrubert, default


params = load_params()
parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=("default", "sequence_window"))
args = parser.parse_args()
profile = args.profile or params.get("pipeline", {}).get("profile", "default")
adapter = can_bigrubert if profile == "sequence_window" else default
adapter.split(params)
