"""Run Kronos outside the Streamlit server process for reliable inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from core import load_predictor, predict, prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--cutoffs", required=True, help="Comma-separated historical cutoffs")
    parser.add_argument("--seeds", required=True, help="Comma-separated fixed seeds")
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top-p", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cutoffs = args.cutoffs.split(",")
    seeds = [int(value) for value in args.seeds.split(",")]
    if not 0 < args.temperature <= 2 or not 0 < args.top_p <= 1:
        raise ValueError("Invalid sampling settings")
    predictor, _, lock = load_predictor()
    paths = []
    for cutoff in cutoffs:
        prepared = prepare(args.symbol, cutoff)
        for seed in seeds:
            forecast = predict(prepared, predictor, lock, seed, args.temperature, args.top_p)
            forecast.insert(0, "seed", seed)
            forecast.insert(0, "cutoff", cutoff)
            paths.append(forecast)
    import pandas as pd
    pd.concat(paths, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
