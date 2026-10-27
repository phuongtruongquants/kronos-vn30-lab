"""Fixed, non-overlapping historical windows for a descriptive seed audit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import HORIZON, eligible_cutoffs, load_calendar, load_stock, prepare, realized_path


AUDIT_START = "2026-01-01"
AUDIT_SEEDS = (42, 43, 44)
AUDIT_WINDOWS = 10
CALIBRATION_WINDOWS = 6


def fixed_cutoffs(symbol: str) -> list[str]:
    """Schedule is determined from the trading calendar, before looking at returns."""
    calendar = load_calendar()
    eligible = set(eligible_cutoffs(symbol))
    first = next((i for i, date in enumerate(calendar) if date >= AUDIT_START), len(calendar))
    frame, _ = load_stock(symbol)
    cutoffs = []
    for i in range(first, len(calendar) - HORIZON, HORIZON):
        date = calendar[i]
        future = calendar[i + 1:i + HORIZON + 1]
        if date in eligible and all(day in frame.index for day in future):
            cutoffs.append(date)
    return cutoffs[-AUDIT_WINDOWS:]


def score(symbol: str, forecasts: pd.DataFrame, cutoffs: list[str], *,
          allow_exploratory: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    if not allow_exploratory:
        raise ValueError("Giá hồi cứu không có bản lưu as-of; chỉ được chấm khi ghi rõ exploratory")
    expected = {(cutoff, seed) for cutoff in cutoffs for seed in AUDIT_SEEDS}
    found = set(zip(forecasts["cutoff"], forecasts["seed"].astype(int)))
    if found != expected or forecasts.duplicated(["cutoff", "seed", "date"]).any():
        raise ValueError("Thiếu, thừa hoặc trùng mốc dự báo/seed")
    trials = []
    for cutoff in cutoffs:
        prepared = prepare(symbol, cutoff)
        actual = realized_path(symbol, cutoff)
        if len(actual) != HORIZON:
            raise ValueError(f"Missing complete future path for {symbol} {cutoff}")
        actual_close = float(actual["close"].iloc[-1])
        prior_close = float(prepared.history["close"].iloc[-1])
        realized = actual_close / prior_close - 1
        for seed, path in forecasts.loc[forecasts.cutoff.eq(cutoff)].groupby("seed"):
            path = path.sort_values("date")
            expected_dates = prepared.target_times.dt.strftime("%Y-%m-%d").tolist()
            if len(path) != HORIZON or path["date"].tolist() != expected_dates:
                raise ValueError(f"Incomplete forecast for {cutoff}, seed {seed}")
            predicted_close = float(path["close"].iloc[-1])
            predicted = predicted_close / prior_close - 1
            trials.append({
                "cutoff": cutoff, "seed": int(seed), "prediction": predicted,
                "realized": realized,
                "hit": int(np.sign(predicted) == np.sign(realized)),
                "absolute_price_error": abs(predicted_close / actual_close - 1),
            })
    panel = pd.DataFrame(trials)
    if panel.empty:
        return panel, panel, None
    calibration_dates = cutoffs[:min(CALIBRATION_WINDOWS, max(0, len(cutoffs) - 2))]
    panel["phase"] = np.where(panel.cutoff.isin(calibration_dates), "Hiệu chỉnh", "Kiểm tra")
    summary = (panel.groupby(["seed", "phase"], as_index=False)
               .agg(wins=("hit", "sum"), windows=("hit", "size"),
                    avg_abs_price_error=("absolute_price_error", "mean")))
    summary["win_rate"] = summary.wins / summary.windows
    chosen = None
    if calibration_dates and len(cutoffs) - len(calibration_dates) >= 2:
        training = summary.loc[summary.phase.eq("Hiệu chỉnh")].sort_values(
            ["win_rate", "avg_abs_price_error", "seed"], ascending=[False, True, True]
        )
        chosen = int(training.seed.iloc[0])
    return panel, summary, chosen
