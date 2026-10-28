"""Separate exploratory DGW replay; never change the fixed 30-stock VN30 panel."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core import (HERE, HORIZON, LOOKBACK, MODEL_REVISION, T, TOP_P,
                  TOKENIZER_REVISION, load_calendar, load_predictor, load_stock,
                  predict, prepare)
from forward import SEEDS, _checkpoint_file, sha256
from retrospective import START, END


SYMBOL = "DGW"
RUN = HERE / "extra_replay_runs" / "DGW_2025-07_to_2026-10"


def build(*, temperature: float = T, destination: Path = RUN,
          paired_baseline: Path | None = None) -> Path:
    if destination.exists():
        raise FileExistsError(f"DGW replay already exists: {destination}")
    calendar = [day for day in load_calendar() if day <= END]
    first = next(i for i, day in enumerate(calendar) if day >= START)
    anchors = [calendar[i] for i in range(first, len(calendar) - HORIZON, HORIZON)]
    frame, metadata = load_stock(SYMBOL)
    available = set(frame.index)
    selected, excluded = [], []
    for cutoff in anchors:
        index = calendar.index(cutoff)
        past = calendar[index - LOOKBACK + 1:index + 1]
        future = calendar[index + 1:index + HORIZON + 1]
        reason = None
        if (len(past) != LOOKBACK or not set(past).issubset(available)
                or (frame.loc[past, "volume"] <= 0).any()):
            reason = "missing_or_invalid_90_bar_lookback"
        elif len(future) != HORIZON or not set(future).issubset(available):
            reason = "missing_10_bar_outcome"
        if reason:
            excluded.append({"cutoff": cutoff, "reason": reason})
        else:
            selected.append(cutoff)
    if not selected:
        raise ValueError("DGW has no complete replay windows")
    destination.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dgw_pending_", dir=destination.parent) as temporary:
        work = Path(temporary)
        pd.DataFrame(excluded, columns=["cutoff", "reason"]).to_csv(work / "exclusions.csv", index=False)
        inputs, prepared = [], []
        for cutoff in selected:
            item = prepare(SYMBOL, cutoff)
            prepared.append(item)
            history = item.history[["date", "open", "high", "low", "close", "volume"]].copy()
            history.insert(0, "cutoff", cutoff)
            inputs.append(history)
        pd.concat(inputs, ignore_index=True).to_csv(work / "inputs.csv", index=False)
        if paired_baseline is not None:
            baseline_manifest = json.loads((paired_baseline / "manifest.json").read_text())
            for name in ("exclusions", "inputs"):
                if sha256(work / f"{name}.csv") != baseline_manifest["file_sha256"][name]:
                    raise ValueError(f"Paired DGW replay changed {name}.csv; aborting")
        predictor, device, lock = load_predictor()
        forecasts = []
        for number, item in enumerate(prepared, 1):
            for seed in SEEDS:
                path = predict(item, predictor, lock, seed, temperature, TOP_P)
                path.insert(0, "seed", seed)
                path.insert(0, "cutoff", item.cutoff)
                forecasts.append(path)
            if number % 10 == 0:
                print(f"DGW: {number}/{len(prepared)} windows", flush=True)
        predictions = pd.concat(forecasts, ignore_index=True)
        predictions.to_csv(work / "forecasts.csv", index=False)

        # Outcome labels are read only after every fixed forecast is on disk.
        outcomes, trials = [], []
        for item in prepared:
            cutoff = item.cutoff
            position = calendar.index(cutoff)
            actual_dates = calendar[position + 1:position + HORIZON + 1]
            expected_dates = pd.bdate_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1),
                                             periods=HORIZON).strftime("%Y-%m-%d").tolist()
            baseline = float(item.history.close.iloc[-1])
            actual_return = float(frame.loc[actual_dates[-1], "close"]) / baseline - 1
            momentum = baseline / float(item.history.close.iloc[-11]) - 1
            mismatch = int(expected_dates != actual_dates)
            outcomes.append({"cutoff": cutoff, "actual_date_10": actual_dates[-1],
                             "forecast_date_10": expected_dates[-1],
                             "last_close": baseline, "actual_return": actual_return,
                             "momentum_10": momentum, "calendar_mismatch": mismatch})
            for seed in SEEDS:
                path = predictions.loc[predictions.cutoff.eq(cutoff) & predictions.seed.eq(seed)].sort_values("date")
                if path.date.tolist() != expected_dates:
                    raise ValueError(f"Incomplete DGW path at {cutoff}, seed {seed}")
                forecast_return = float(path.close.iloc[-1]) / baseline - 1
                trials.append({"cutoff": cutoff, "seed": seed,
                               "forecast_return": forecast_return, "actual_return": actual_return,
                               "hit": int(np.sign(forecast_return) == np.sign(actual_return)),
                               "momentum_hit": int(np.sign(momentum) == np.sign(actual_return)),
                               "always_up_hit": int(actual_return > 0),
                               "calendar_mismatch": mismatch})
        pd.DataFrame(outcomes).to_csv(work / "outcomes.csv", index=False)
        if paired_baseline is not None:
            if sha256(work / "outcomes.csv") != baseline_manifest["file_sha256"]["outcomes"]:
                raise ValueError("Paired DGW replay changed outcomes.csv; aborting")
        panel = pd.DataFrame(trials)
        panel.to_csv(work / "panel.csv", index=False)
        summary = panel.groupby("seed", as_index=False).agg(
            correct=("hit", "sum"), samples=("hit", "size"),
            momentum_correct=("momentum_hit", "sum"), always_up_correct=("always_up_hit", "sum"))
        summary["win_rate"] = summary.correct / summary.samples
        summary["momentum_rate"] = summary.momentum_correct / summary.samples
        summary["always_up_rate"] = summary.always_up_correct / summary.samples
        summary.to_csv(work / "summary.csv", index=False)
        manifest = {
            "kind": "exploratory_single_stock_not_point_in_time", "symbol": SYMBOL,
            "created_at_ict": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
            "start": START, "end": END, "anchors": anchors,
            "selected_windows": len(selected), "excluded_windows": len(excluded),
            "source": metadata["source"], "source_retrieved_at_utc": metadata["retrieved_at_utc"],
            "data_sha256": sha256(HERE / "extra_stock_data" / "DGW.csv"),
            "model_revision": MODEL_REVISION, "tokenizer_revision": TOKENIZER_REVISION,
            "model_sha256": sha256(_checkpoint_file("NeoQuasar/Kronos-small", MODEL_REVISION)),
            "tokenizer_sha256": sha256(_checkpoint_file("NeoQuasar/Kronos-Tokenizer-base", TOKENIZER_REVISION)),
            "seeds": list(SEEDS), "temperature": temperature, "top_p": TOP_P,
            "paired_baseline": str(paired_baseline) if paired_baseline else None,
            "lookback": LOOKBACK, "horizon": HORIZON, "device": device,
            "source_sha256": {"core.py": sha256(HERE / "core.py"),
                              "extra_replay.py": sha256(HERE / "extra_replay.py")},
            "file_sha256": {name: sha256(work / f"{name}.csv") for name in
                            ("exclusions", "inputs", "forecasts", "outcomes", "panel", "summary")},
        }
        (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        work.rename(destination)
    return destination


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=T)
    parser.add_argument("--destination", type=Path, default=RUN)
    parser.add_argument("--paired-baseline", type=Path)
    args = parser.parse_args()
    print(build(temperature=args.temperature, destination=args.destination,
                paired_baseline=args.paired_baseline))
