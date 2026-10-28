"""Create a reproducible, standalone DGW snapshot from the existing VCI feed."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from refresh_latest import URL, decode


THROUGH = "2026-10-27"
HERE = Path(__file__).resolve().parent
DESTINATION = HERE / "extra_stock_data"


def main() -> None:
    cutoff = datetime.fromisoformat(THROUGH).date()
    next_midnight = datetime.combine(cutoff + timedelta(days=1), datetime.min.time(),
                                     tzinfo=timezone(timedelta(hours=7)))
    response = requests.post(URL, json={"timeFrame": "ONE_DAY", "symbols": ["DGW"],
                                        "to": int(next_midnight.timestamp()), "countBack": 3000},
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://trading.vietcap.com.vn/",
                                      "Origin": "https://trading.vietcap.com.vn",
                                      "Content-Type": "application/json"}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if len(payload) != 1 or payload[0].get("symbol") != "DGW":
        raise ValueError("VCI did not return one DGW series")
    bars = decode(payload[0]).sort_index()
    if bars.empty or bars.index[-1] != THROUGH or bars.index.has_duplicates:
        raise ValueError("DGW history missing final session or has duplicate dates")
    values = bars[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if (not np.isfinite(values).all()
            or (bars[["open", "high", "low", "close"]] <= 0).any().any()
            or (bars.volume < 0).any()
            or (bars.high < bars[["open", "close"]].max(axis=1)).any()
            or (bars.low > bars[["open", "close"]].min(axis=1)).any()):
        raise ValueError("DGW source contains invalid OHLCV")
    output = bars.reset_index().rename(columns={"date": "time"})
    output["time"] += " 07:00:00+07:00"
    content = output.to_csv(index=False).encode()
    DESTINATION.mkdir(exist_ok=True)
    (DESTINATION / "DGW.csv").write_bytes(content)
    metadata = {
        "symbol": "DGW", "rows": len(output),
        "first_date": str(bars.index[0]), "last_date": THROUGH,
        "source": "Vietcap trading chart OHLCChart/gap-chart via direct HTTPS",
        "source_url": URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_unit": "thousand VND; VCI integer price / 1000",
        "adjustment": "Current source snapshot; historical as-of vintages unavailable",
        "zero_volume_bars": int((bars.volume == 0).sum()),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    (DESTINATION / "DGW.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"DGW: {len(output)} bars, {bars.index[0]}..{THROUGH}, close={bars.iloc[-1]['close']:.2f}")


if __name__ == "__main__":
    main()
