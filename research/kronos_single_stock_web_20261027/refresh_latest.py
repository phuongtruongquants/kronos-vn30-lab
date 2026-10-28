"""Fetch a completed session for the 30 VN30 stocks and separately labeled DGW."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from core import EXTRA_STOCKS, current_members, load_stock


HERE = Path(__file__).resolve().parent
URL = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"


def decode(raw: dict) -> pd.DataFrame:
    fields = ("t", "o", "h", "l", "c", "v")
    if any(key not in raw for key in fields) or len({len(raw[key]) for key in fields}) != 1:
        raise ValueError("Malformed Vietcap OHLCV response")
    dates = [datetime.fromtimestamp(int(stamp), timezone.utc).date().isoformat() for stamp in raw["t"]]
    frame = pd.DataFrame({
        "date": dates,
        "open": np.asarray(raw["o"], dtype=float) / 1000,
        "high": np.asarray(raw["h"], dtype=float) / 1000,
        "low": np.asarray(raw["l"], dtype=float) / 1000,
        "close": np.asarray(raw["c"], dtype=float) / 1000,
        "volume": np.asarray(raw["v"], dtype=np.int64),
    })
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate source date")
    return frame.set_index("date")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Completed session, YYYY-MM-DD")
    args = parser.parse_args()
    target = datetime.strptime(args.date, "%Y-%m-%d").date()
    if target >= datetime.now(timezone(timedelta(hours=7))).date():
        # After the close, the current session can be available. Require explicit date,
        # then verify all requested bars below rather than assuming the feed is final.
        print("Checking current-session bars explicitly; verify again after final close if needed.")
    to_stamp = int(datetime.combine(target + timedelta(days=2), datetime.min.time(),
                                    tzinfo=timezone(timedelta(hours=7))).timestamp())
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://trading.vietcap.com.vn/",
        "Origin": "https://trading.vietcap.com.vn",
        "Content-Type": "application/json",
    })
    members = current_members() + list(EXTRA_STOCKS)
    rows, checks = [], []
    for number, symbol in enumerate(members, 1):
        response = session.post(URL, json={"timeFrame": "ONE_DAY", "symbols": [symbol],
                                           "to": to_stamp, "countBack": 6}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if len(payload) != 1 or payload[0].get("symbol") != symbol:
            raise ValueError(f"No single-symbol bars for {symbol}")
        recent = decode(payload[0])
        known, _ = load_stock(symbol)
        past_known = known.loc[known.index < args.date]
        if past_known.empty:
            raise ValueError(f"{symbol} has no session before {args.date}")
        previous = past_known.iloc[-1]
        previous_date = past_known.index[-1]
        if previous_date not in recent.index or args.date not in recent.index:
            raise ValueError(f"{symbol} missing overlap {previous_date} or target {args.date}")
        source_previous = recent.loc[previous_date]
        relative = max(abs(float(previous[field]) / float(source_previous[field]) - 1)
                       for field in ("open", "high", "low", "close"))
        volume_equal = int(previous["volume"]) == int(source_previous["volume"])
        if relative > .001 or not volume_equal:
            raise ValueError(f"{symbol} overlap changed: {relative:.4%}, volume equal={volume_equal}")
        bar = recent.loc[args.date]
        values = bar[["open", "high", "low", "close", "volume"]].to_numpy(float)
        if (not np.isfinite(values).all() or (values <= 0).any()
                or bar["high"] < max(bar["open"], bar["close"])
                or bar["low"] > min(bar["open"], bar["close"])):
            raise ValueError(f"Invalid target bar for {symbol}")
        rows.append({"symbol": symbol, "time": f"{args.date} 07:00:00+07:00", **bar.to_dict()})
        checks.append({"symbol": symbol, "overlap_date": previous_date,
                       "overlap_max_relative_difference": relative,
                       "overlap_volume_equal": volume_equal,
                       "source_latest_date": recent.index[-1]})
        print(f"[{number}/{len(members)}] {symbol}: {bar['close']:.3f}, overlap {relative:.4%}", flush=True)
        time.sleep(.15)

    output = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    if len(output) != len(members) or output.symbol.nunique() != len(members):
        raise ValueError(f"Expected exactly {len(members)} validated bars")
    directory = HERE / f"data_update_{target:%Y%m%d}"
    directory.mkdir(parents=True, exist_ok=True)
    csv_bytes = output.to_csv(index=False).encode("utf-8")
    (directory / f"ohlcv_{args.date}.csv").write_bytes(csv_bytes)
    manifest = {
        "target_session": args.date,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Vietcap trading chart OHLCChart/gap-chart via direct HTTPS",
        "source_url": URL,
        "price_unit": "thousand VND; VCI integer price / 1000",
        "adjustment_check": "Prior session OHLCV checked against bundled adjusted history",
        "symbols": members,
        "rows": len(output),
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "checks": checks,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(output)} validated bars to {directory}")


if __name__ == "__main__":
    main()
