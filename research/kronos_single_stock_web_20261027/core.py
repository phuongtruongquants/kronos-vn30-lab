"""One-stock Kronos inference, isolated from any alpha or portfolio rule."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "vn30_long_20261026" / "data"
MEMBERSHIP = HERE.parent / "vn30_long_20261026" / "membership_intervals_2018_2026.csv"
UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
LOOKBACK = 90
HORIZON = 10
T = 0.6
TOP_P = 0.9
SAMPLE_COUNT = 1


@dataclass
class Prepared:
    symbol: str
    cutoff: str
    history: pd.DataFrame
    target_times: pd.Series
    source_metadata: dict


def current_members() -> list[str]:
    intervals = pd.read_csv(MEMBERSHIP, dtype=str).fillna("")
    last_start = intervals["analysis_period_start"].max()
    members = intervals.loc[
        (intervals["analysis_period_start"] <= last_start)
        & ((intervals["analysis_period_end_exclusive"] == "")
           | (intervals["analysis_period_end_exclusive"] > last_start)),
        "ticker",
    ]
    symbols = sorted(members.unique().tolist())
    if len(symbols) != 30:
        raise ValueError(f"Expected 30 latest VN30 members, found {len(symbols)}")
    return symbols


def updates() -> list[tuple[str, Path, dict]]:
    """Validated daily overlays, sorted by session; never rewrite the research cache."""
    result = []
    for manifest_path in HERE.glob("data_update_*/manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        date = manifest["target_session"]
        csv = manifest_path.parent / f"ohlcv_{date}.csv"
        if csv.exists():
            result.append((date, csv, manifest))
    return sorted(result)


def load_stock(symbol: str, *, allow_historical_member: bool = False) -> tuple[pd.DataFrame, dict]:
    if not allow_historical_member and symbol not in current_members():
        raise ValueError("Mã không thuộc danh sách 30 thành viên cuối trong dữ liệu nghiên cứu")
    path = DATA_ROOT / f"{symbol}.csv"
    if not path.exists():
        raise ValueError(f"Chưa có tệp giá cho mã {symbol}")
    frame = pd.read_csv(path)
    latest_update = None
    for date, csv, manifest in updates():
        update = pd.read_csv(csv)
        update = update.loc[update["symbol"].eq(symbol)].drop(columns="symbol")
        if len(update) != 1:
            raise ValueError(f"Thiếu nến {symbol} ngày {date} trong dữ liệu cập nhật")
        if str(frame["time"].iloc[-1])[:10] < date:
            frame = pd.concat([frame, update], ignore_index=True)
            latest_update = manifest
    required = {"time", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame):
        raise ValueError(f"Thiếu cột dữ liệu: {required - set(frame)}")
    frame["time"] = pd.to_datetime(frame["time"])
    frame["date"] = frame["time"].dt.strftime("%Y-%m-%d")
    frame = frame.sort_values("time").drop_duplicates("date", keep=False).set_index("date", drop=False)
    metadata_path = DATA_ROOT / f"{symbol}.metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if latest_update is not None and frame.index[-1] == latest_update["target_session"]:
        metadata = {**metadata, "retrieved_at_utc": latest_update["retrieved_at_utc"],
                    "latest_source": latest_update["source"], "last_date": latest_update["target_session"]}
    return frame, metadata


def load_calendar() -> list[str]:
    calendar = pd.read_csv(DATA_ROOT / "VN30_calendar.csv")
    dates = calendar["time"].astype(str).tolist()
    dates.extend(date for date, _, _ in updates())
    return sorted(set(dates))


def eligible_cutoffs(symbol: str) -> list[str]:
    frame, _ = load_stock(symbol)
    sessions = load_calendar()
    valid = set(frame.index)
    return [sessions[i] for i in range(LOOKBACK - 1, len(sessions))
            if all(date in valid for date in sessions[i - LOOKBACK + 1 : i + 1])]


def prepare(symbol: str, cutoff: str) -> Prepared:
    frame, metadata = load_stock(symbol)
    sessions = load_calendar()
    if cutoff not in sessions:
        raise ValueError("Ngày chốt không có trong lịch phiên của bộ dữ liệu")
    pos = sessions.index(cutoff)
    history_dates = sessions[pos - LOOKBACK + 1 : pos + 1]
    if len(history_dates) != LOOKBACK or any(date not in frame.index for date in history_dates):
        raise ValueError("Mã này không đủ 90 phiên liên tục trước ngày chốt")
    history = frame.loc[history_dates].copy()
    values = history[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Dữ liệu nến đầu vào chứa giá hoặc khối lượng không hợp lệ")
    if ((history["low"] > history[["open", "close"]].min(axis=1)).any()
            or (history["high"] < history[["open", "close"]].max(axis=1)).any()):
        raise ValueError("OHLC đầu vào không nhất quán")

    # Forecast dates must be knowable at cutoff. Never pass the realized future
    # exchange calendar (which can include unannounced closures) to the model.
    future_dates = pd.bdate_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1),
                                  periods=HORIZON).strftime("%Y-%m-%d").tolist()
    target_times = pd.Series(pd.to_datetime([f"{d} 07:00:00+07:00" for d in future_dates]))
    return Prepared(symbol, cutoff, history, target_times, metadata)


def realized_path(symbol: str, cutoff: str) -> pd.DataFrame:
    """Evaluation only. This function is never called by the inference worker."""
    frame, _ = load_stock(symbol, allow_historical_member=True)
    sessions = load_calendar()
    if cutoff not in sessions:
        raise ValueError("Ngày chốt không có trong lịch phiên")
    dates = [date for date in sessions[sessions.index(cutoff) + 1:] if date in frame.index]
    return frame.loc[dates[:HORIZON]].copy() if dates else frame.iloc[0:0].copy()


def locate_upstream() -> Path:
    vendored = HERE.parent.parent / "vendor" / "Kronos"
    if (vendored / "model" / "kronos.py").exists():
        if (vendored / "UPSTREAM_COMMIT").read_text().strip() != UPSTREAM_COMMIT:
            raise RuntimeError("Bản Kronos đóng gói sai phiên bản")
        return vendored
    options = [
        Path(os.environ.get("KRONOS_ROOT", "/tmp/vn30_kronos_vinagent_20261026/Kronos")),
        Path.home() / ".cache" / "kronos-single-stock" / "Kronos",
    ]
    for path in options:
        if (path / "model" / "kronos.py").exists():
            actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            if actual == UPSTREAM_COMMIT:
                return path
    destination = options[-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError("Bản Kronos cache sai phiên bản; hãy đặt KRONOS_ROOT trỏ đến commit đã khóa")
    subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/shiyu-coder/Kronos.git", str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", UPSTREAM_COMMIT], check=True)
    return destination


def load_predictor():
    import threading
    import torch

    root = locate_upstream()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()
    # Streamlit runs inference on a worker thread; MPS has crashed this host in
    # that setting. CPU is the stable local default; opt into MPS explicitly.
    device = os.environ.get("KRONOS_DEVICE", "cpu")
    if device not in {"cpu", "mps"}:
        raise ValueError("KRONOS_DEVICE must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this host")
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    return predictor, device, threading.Lock()


def predict(prepared: Prepared, predictor, lock: threading.Lock, seed: int = 42,
            temperature: float = T, top_p: float = TOP_P) -> pd.DataFrame:
    import torch

    with lock, torch.inference_mode():
        torch.manual_seed(seed)
        np.random.seed(seed)
        output = predictor.predict(
            prepared.history[["open", "high", "low", "close", "volume"]],
            pd.Series(prepared.history["time"].tolist()),
            prepared.target_times,
            pred_len=HORIZON,
            T=temperature,
            top_p=top_p,
            sample_count=1,
            verbose=False,
        )
    if len(output) != HORIZON or not np.isfinite(output[["open", "high", "low", "close"]].to_numpy(float)).all():
        raise ValueError("Kronos trả về dự báo không hợp lệ")
    result = output.reset_index(drop=True).copy()
    result.insert(0, "date", prepared.target_times.dt.strftime("%Y-%m-%d"))
    return result
