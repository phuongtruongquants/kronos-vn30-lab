"""Seal forecasts before outcomes exist, then score them in a later run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core import (HERE, HORIZON, LOOKBACK, MODEL_REVISION, T, TOP_P,
                  TOKENIZER_REVISION, current_members, load_predictor,
                  load_stock, predict, prepare, realized_path)


LEDGER = HERE / "forward_runs"
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_file(repository: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repository, "model.safetensors", revision=revision))


def freeze() -> Path:
    """One predetermined VN30 run. Refuse stale data or existing run directory."""
    import torch

    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    symbols = current_members()
    latest = {symbol: load_stock(symbol)[0].index[-1] for symbol in symbols}
    if len(set(latest.values())) != 1 or next(iter(latest.values())) != now.date().isoformat():
        raise ValueError("Chỉ được niêm phong khi đủ 30 mã có giá của phiên hôm nay")
    if (now.hour, now.minute) < (15, 0):
        raise ValueError("Chờ hết phiên giao dịch trước khi niêm phong")
    cutoff = now.date().isoformat()
    destination = LEDGER / cutoff
    if destination.exists():
        raise FileExistsError(f"Lô {cutoff} đã tồn tại; không được ghi đè")

    import tempfile
    LEDGER.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pending_", dir=LEDGER) as temporary:
        work = Path(temporary)
        inputs = []
        prepared_by_symbol = {}
        for symbol in symbols:
            prepared = prepare(symbol, cutoff)
            prepared_by_symbol[symbol] = prepared
            input_frame = prepared.history[["date", "open", "high", "low", "close", "volume"]].copy()
            input_frame.insert(0, "symbol", symbol)
            inputs.append(input_frame)
        pd.concat(inputs, ignore_index=True).to_csv(work / "inputs.csv", index=False)

        predictor, device, lock = load_predictor()
        paths = []
        for index, symbol in enumerate(symbols, start=1):
            prepared = prepared_by_symbol[symbol]
            for seed in SEEDS:
                forecast = predict(prepared, predictor, lock, seed, T, TOP_P)
                forecast.insert(0, "seed", seed)
                forecast.insert(0, "cutoff", cutoff)
                forecast.insert(0, "symbol", symbol)
                paths.append(forecast)
            print(f"{index}/{len(symbols)} {symbol}", flush=True)
        pd.concat(paths, ignore_index=True).to_csv(work / "forecasts.csv", index=False)
        manifest = {
            "sealed_at_ict": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
            "cutoff": cutoff, "symbols": symbols, "seeds": list(SEEDS),
            "lookback": LOOKBACK, "horizon": HORIZON,
            "temperature": T, "top_p": TOP_P, "device": device,
            "model_revision": MODEL_REVISION, "tokenizer_revision": TOKENIZER_REVISION,
            "model_sha256": sha256(_checkpoint_file("NeoQuasar/Kronos-small", MODEL_REVISION)),
            "tokenizer_sha256": sha256(_checkpoint_file("NeoQuasar/Kronos-Tokenizer-base", TOKENIZER_REVISION)),
            "git_head_at_freeze": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {
                "core.py": sha256(HERE / "core.py"),
                "forward.py": sha256(HERE / "forward.py"),
                "vendor/Kronos/model/kronos.py": sha256(HERE.parent.parent / "vendor/Kronos/model/kronos.py"),
            },
            "python": platform.python_version(), "torch": torch.__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
            "calendar_rule": "weekdays after cutoff; actual outcome is ordinal trading session 10",
            "universe_rule": "30 members in 2026-08-03 published VN30 basket",
            "inputs_sha256": sha256(work / "inputs.csv"),
            "forecasts_sha256": sha256(work / "forecasts.csv"),
        }
        (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        work.rename(destination)
    return destination


def evaluate(run: Path) -> pd.DataFrame:
    """No score until each symbol has 10 subsequent observed sessions."""
    manifest = json.loads((run / "manifest.json").read_text())
    if datetime.fromisoformat(manifest["sealed_at_ict"]).date().isoformat() != manifest["cutoff"]:
        raise ValueError("Lô không được niêm phong trong ngày chốt")
    for name in ("inputs", "forecasts"):
        if sha256(run / f"{name}.csv") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Ledger {name} checksum mismatch")
    forecasts = pd.read_csv(run / "forecasts.csv")
    inputs = pd.read_csv(run / "inputs.csv")
    expected = len(manifest["symbols"]) * len(manifest["seeds"]) * manifest["horizon"]
    if len(forecasts) != expected or forecasts.duplicated(["symbol", "seed", "date"]).any():
        raise ValueError("Ledger thiếu hoặc trùng đường dự báo")
    grid = {(symbol, seed) for symbol in manifest["symbols"] for seed in manifest["seeds"]}
    found = set(zip(forecasts["symbol"], forecasts["seed"].astype(int)))
    if found != grid or set(forecasts["cutoff"]) != {manifest["cutoff"]}:
        raise ValueError("Ledger có mã, seed hoặc ngày chốt ngoài cấu hình")
    if (inputs.groupby("symbol").size().to_dict()
            != {symbol: manifest["lookback"] for symbol in manifest["symbols"]}):
        raise ValueError("Ledger đầu vào không đủ 90 nến mỗi mã")
    if (inputs["date"] > manifest["cutoff"]).any():
        raise ValueError("Ledger đầu vào chứa nến sau ngày chốt")
    rows = []
    for symbol in manifest["symbols"]:
        for seed in manifest["seeds"]:
            if len(forecasts.loc[forecasts.symbol.eq(symbol) & forecasts.seed.eq(seed)]) != manifest["horizon"]:
                raise ValueError(f"Thiếu dự báo {symbol} seed {seed}")
        current, _ = load_stock(symbol, allow_historical_member=True)
        sealed = inputs.loc[inputs.symbol.eq(symbol)].iloc[-1]
        if manifest["cutoff"] not in current.index:
            raise ValueError(f"Thiếu giá ngày chốt của {symbol}")
        for field in ("open", "high", "low", "close", "volume"):
            if not np.isclose(float(current.loc[manifest["cutoff"], field]),
                              float(sealed[field]), rtol=1e-9, atol=1e-7):
                raise ValueError(f"Giá ngày chốt {symbol} đã bị điều chỉnh; cần xử lý corporate action trước khi chấm")
        actual = realized_path(symbol, manifest["cutoff"])
        if len(actual) < manifest["horizon"]:
            continue
        forecast_dates = forecasts.loc[forecasts.symbol.eq(symbol) & forecasts.seed.eq(manifest["seeds"][0]), "date"].tolist()
        if actual["date"].tolist() != forecast_dates:
            raise ValueError("Lịch phiên thực tế khác lịch dự báo biết trước; lô này không được chấm")
        baseline = inputs.loc[inputs.symbol.eq(symbol), "close"].iloc[-1]
        actual_return = actual["close"].iloc[-1] / baseline - 1
        for seed in manifest["seeds"]:
            path = forecasts.loc[forecasts.symbol.eq(symbol) & forecasts.seed.eq(seed)]
            if len(path) != manifest["horizon"]:
                raise ValueError(f"Thiếu dự báo {symbol} seed {seed}")
            forecast_return = path.iloc[-1]["close"] / baseline - 1
            rows.append({"symbol": symbol, "seed": seed, "forecast_return": forecast_return,
                         "actual_return": actual_return,
                         "hit": int(np.sign(forecast_return) == np.sign(actual_return))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze())
    else:
        for path in ([args.run] if args.run else sorted(LEDGER.glob("20*"))):
            print(path, evaluate(path).to_string(index=False))
