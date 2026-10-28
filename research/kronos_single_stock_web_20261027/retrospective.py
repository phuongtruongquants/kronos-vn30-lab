"""Exploratory replay on the 2026-retrieved VCI/Vnstock price snapshot.

This is deliberately separate from forward.py: past price vintages are not
available, so these results must never be called leakage-free out of sample.
"""

from __future__ import annotations

import json
import platform
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core import (HERE, HORIZON, LOOKBACK, MODEL_REVISION, T, TOP_P,
                  TOKENIZER_REVISION, current_members, load_calendar,
                  load_predictor, load_stock, predict, prepare)
from forward import SEEDS, _checkpoint_file, sha256


START = "2025-07-01"
END = "2026-10-27"
RUN = HERE / "retrospective_runs" / "2025-07_to_2026-10"


def plan() -> tuple[list[tuple[str, str]], pd.DataFrame, dict[str, object]]:
    calendar = [day for day in load_calendar() if day <= END]
    first = next(i for i, day in enumerate(calendar) if day >= START)
    anchors = [calendar[i] for i in range(first, len(calendar) - HORIZON, HORIZON)]
    selected: list[tuple[str, str]] = []
    excluded = []
    for symbol in current_members():
        frame, _ = load_stock(symbol)
        available = set(frame.index)
        for cutoff in anchors:
            index = calendar.index(cutoff)
            past = calendar[index - LOOKBACK + 1:index + 1]
            future = calendar[index + 1:index + HORIZON + 1]
            reason = None
            if len(past) != LOOKBACK or not set(past).issubset(available):
                reason = "missing_90_bar_lookback"
            elif len(future) != HORIZON or not set(future).issubset(available):
                reason = "missing_10_bar_outcome"
            if reason:
                excluded.append({"symbol": symbol, "cutoff": cutoff, "reason": reason})
            else:
                selected.append((symbol, cutoff))
    return selected, pd.DataFrame(excluded), {"anchors": anchors, "calendar_end": calendar[-1]}


def _score(work: Path, selected: list[tuple[str, str]]) -> None:
    """Join realized labels only after all fixed forecasts have been saved."""
    predictions = pd.read_csv(work / "forecasts.csv")
    inputs = pd.read_csv(work / "inputs.csv")
    expected = {(symbol, cutoff, seed) for symbol, cutoff in selected for seed in SEEDS}
    groups = predictions.groupby(["symbol", "cutoff", "seed"]).size()
    if set(groups.index) != expected or not groups.eq(HORIZON).all():
        raise ValueError("Thiếu hoặc thừa mã/ngày/seed trong dự báo hồi cứu")
    if predictions.duplicated(["symbol", "cutoff", "seed", "date"]).any():
        raise ValueError("Trùng ngày trong đường dự báo")
    if (inputs["date"] > inputs["cutoff"]).any():
        raise ValueError("Có nến đầu vào sau ngày chốt")

    stock_frames = {symbol: load_stock(symbol)[0] for symbol in current_members()}
    calendar = [day for day in load_calendar() if day <= END]
    actual_rows, trials = [], []
    for symbol, cutoff in selected:
        frame = stock_frames[symbol]
        pos = calendar.index(cutoff)
        actual_dates = calendar[pos + 1:pos + HORIZON + 1]
        history = inputs.loc[inputs.symbol.eq(symbol) & inputs.cutoff.eq(cutoff)]
        if len(history) != LOOKBACK or not history["date"].is_monotonic_increasing:
            raise ValueError(f"Lỗi snapshot đầu vào {symbol} {cutoff}")
        if not np.isclose(float(history.iloc[-1]["close"]), float(frame.loc[cutoff, "close"])):
            raise ValueError(f"Giá đầu vào và kết quả khác cơ sở điều chỉnh: {symbol} {cutoff}")
        baseline = float(history.iloc[-1]["close"])
        realized_close = float(frame.loc[actual_dates[-1], "close"])
        actual_return = realized_close / baseline - 1
        momentum = baseline / float(history.iloc[-11]["close"]) - 1
        expected_dates = pd.bdate_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1),
                                         periods=HORIZON).strftime("%Y-%m-%d").tolist()
        calendar_mismatch = int(expected_dates != actual_dates)
        actual_rows.append({"symbol": symbol, "cutoff": cutoff,
                            "forecast_date_10": expected_dates[-1],
                            "actual_date_10": actual_dates[-1], "actual_close_10": realized_close,
                            "last_close": baseline, "actual_return": actual_return,
                            "momentum_10": momentum, "calendar_mismatch": calendar_mismatch})
        for seed in SEEDS:
            path = predictions.loc[predictions.symbol.eq(symbol) & predictions.cutoff.eq(cutoff)
                                   & predictions.seed.eq(seed)].sort_values("date")
            if path["date"].tolist() != expected_dates:
                raise ValueError(f"Sai lịch dự báo {symbol} {cutoff} seed {seed}")
            forecast_close = float(path.iloc[-1]["close"])
            forecast_return = forecast_close / baseline - 1
            trials.append({"symbol": symbol, "cutoff": cutoff, "seed": seed,
                           "forecast_return": forecast_return, "actual_return": actual_return,
                           "calendar_mismatch": calendar_mismatch,
                           "hit": int(np.sign(forecast_return) == np.sign(actual_return)),
                           "momentum_hit": int(np.sign(momentum) == np.sign(actual_return)),
                           "always_up_hit": int(actual_return > 0),
                           "abs_return_error": abs(forecast_return - actual_return)})
    pd.DataFrame(actual_rows).to_csv(work / "outcomes.csv", index=False)
    panel = pd.DataFrame(trials)
    panel.to_csv(work / "panel.csv", index=False)
    summary = panel.groupby("seed", as_index=False).agg(
        correct=("hit", "sum"), samples=("hit", "size"),
        momentum_correct=("momentum_hit", "sum"), always_up_correct=("always_up_hit", "sum"),
        mean_abs_return_error=("abs_return_error", "mean"))
    summary["win_rate"] = summary.correct / summary.samples
    summary["momentum_rate"] = summary.momentum_correct / summary.samples
    summary["always_up_rate"] = summary.always_up_correct / summary.samples
    summary.to_csv(work / "summary.csv", index=False)
    per_stock = panel.groupby(["symbol", "seed"], as_index=False).agg(
        correct=("hit", "sum"), samples=("hit", "size"),
        momentum_correct=("momentum_hit", "sum"), always_up_correct=("always_up_hit", "sum"))
    per_stock["win_rate"] = per_stock.correct / per_stock.samples
    per_stock.to_csv(work / "per_stock.csv", index=False)
    panel.groupby(["cutoff", "seed"], as_index=False).agg(
        hit_rate=("hit", "mean"), momentum_rate=("momentum_hit", "mean"),
        always_up_rate=("always_up_hit", "mean"), symbols=("hit", "size")
    ).to_csv(work / "per_date.csv", index=False)


def build() -> Path:
    import torch

    if RUN.exists():
        raise FileExistsError(f"Lô hồi cứu đã có: {RUN}")
    selected, excluded, schedule = plan()
    if not selected:
        raise ValueError("Không có mốc hợp lệ")
    RUN.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="retro_pending_", dir=RUN.parent) as temporary:
        work = Path(temporary)
        excluded.to_csv(work / "exclusions.csv", index=False)
        pd.DataFrame(selected, columns=["symbol", "cutoff"]).to_csv(work / "plan.csv", index=False)
        prepared_items = []
        input_parts = []
        for symbol, cutoff in selected:
            item = prepare(symbol, cutoff)
            prepared_items.append(item)
            bars = item.history[["date", "open", "high", "low", "close", "volume"]].copy()
            bars.insert(0, "cutoff", cutoff)
            bars.insert(0, "symbol", symbol)
            input_parts.append(bars)
        pd.concat(input_parts, ignore_index=True).to_csv(work / "inputs.csv", index=False)
        predictor, device, lock = load_predictor()
        forecasts = []
        done = 0
        for symbol in current_members():
            for item in (item for item in prepared_items if item.symbol == symbol):
                for seed in SEEDS:
                    path = predict(item, predictor, lock, seed, T, TOP_P)
                    path.insert(0, "seed", seed)
                    path.insert(0, "cutoff", item.cutoff)
                    path.insert(0, "symbol", symbol)
                    forecasts.append(path)
                    done += 1
            print(f"{symbol}: {done}/{len(selected) * len(SEEDS)} paths", flush=True)
        pd.concat(forecasts, ignore_index=True).to_csv(work / "forecasts.csv", index=False)
        _score(work, selected)
        manifest = {
            "kind": "exploratory_retrospective_not_point_in_time",
            "created_at_ict": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
            "start": START, "end": END, "anchors": schedule["anchors"],
            "calendar_end": schedule["calendar_end"],
            "universe": "30 stocks in August 2026 VN30 basket, including pre-membership history",
            "source": "Vietcap VCI via Vnstock, retrieved 2026-10-26 with 26/27 Oct overlays",
            "adjustment": "retrospectively adjusted; historical as-of vintages unavailable",
            "model_revision": MODEL_REVISION, "tokenizer_revision": TOKENIZER_REVISION,
            "model_sha256": sha256(_checkpoint_file("NeoQuasar/Kronos-small", MODEL_REVISION)),
            "tokenizer_sha256": sha256(_checkpoint_file("NeoQuasar/Kronos-Tokenizer-base", TOKENIZER_REVISION)),
            "seeds": list(SEEDS), "temperature": T, "top_p": TOP_P,
            "lookback": LOOKBACK, "horizon": HORIZON, "device": device,
            "python": platform.python_version(), "torch": torch.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
            "selected_pairs": len(selected), "excluded_pairs": len(excluded),
            "source_sha256": {"core.py": sha256(HERE / "core.py"),
                              "retrospective.py": sha256(HERE / "retrospective.py")},
            "file_sha256": {name: sha256(work / f"{name}.csv") for name in
                            ("plan", "exclusions", "inputs", "forecasts", "outcomes",
                             "panel", "summary", "per_stock", "per_date")},
        }
        (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        work.rename(RUN)
    return RUN


if __name__ == "__main__":
    print(build())
