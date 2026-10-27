"""Run Kronos outside the Streamlit server process for reliable inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from core import load_predictor, predict, prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepared = prepare(args.symbol, args.cutoff)
    predictor, _, lock = load_predictor()
    forecast = predict(prepared, predictor, lock, args.seed)
    forecast.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
