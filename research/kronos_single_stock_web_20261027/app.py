"""Local interactive Kronos forecast viewer for one VN30 stock at a time."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from chart import chart_html

from core import (
    HORIZON, LOOKBACK, MODEL_REVISION, SAMPLE_COUNT, T, TOP_P,
    current_members, eligible_cutoffs, load_stock, prepare,
)


st.set_page_config(page_title="Kronos Lab · VN30", page_icon="📈", layout="wide")
st.markdown("""
<style>
  .stApp { background: #f5f7f6; color: #15231f; }
  .block-container { max-width: 1420px; padding-top: 2.5rem; }
  h1, h2, h3 { color: #12362d; letter-spacing: -.035em; }
  h1 { font-size: clamp(2.5rem, 5vw, 4.4rem); line-height: 1; }
  [data-testid="stMetric"] { background: white; border: 1px solid #dce5df; border-radius: 16px; padding: 16px; }
  [data-testid="stSidebar"] { background: #e9f0eb; }
  [data-testid="stBaseButton-primary"] { background: #0b8064; border-color: #0b8064; }
  [data-testid="stBaseButton-primary"]:hover { background: #075c49; border-color: #075c49; }
  .eyebrow { color: #17735a; text-transform: uppercase; font-size: .78rem; font-weight: 750; letter-spacing: .16em; }
  .subtle { color: #587067; font-size: 1rem; }
  .note { background: #e7f2ed; border-left: 4px solid #168365; padding: 12px 16px; border-radius: 8px; color: #254b3f; }
</style>
""", unsafe_allow_html=True)


def run_kronos(symbol: str, cutoff: str, seed: int) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="kronos_single_stock_") as directory:
        output = Path(directory) / "forecast.csv"
        env = os.environ.copy()
        env["KRONOS_DEVICE"] = "cpu"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "worker.py"),
             "--symbol", symbol, "--cutoff", cutoff, "--seed", str(seed), "--output", str(output)],
            capture_output=True, text=True, timeout=180, env=env,
        )
        if completed.returncode != 0 or not output.exists():
            detail = completed.stderr[-1200:] or completed.stdout[-1200:]
            raise RuntimeError(detail or f"Kronos worker exited {completed.returncode}")
        return pd.read_csv(output)


st.markdown('<div class="eyebrow">Thử riêng mô hình · không có chiến lược giao dịch</div>', unsafe_allow_html=True)
st.title("Kronos Lab")
st.markdown('<div class="subtle">Chọn một cổ phiếu VN30 và xem Kronos dự báo 10 phiên tiếp theo từ 90 nến đã biết.</div>', unsafe_allow_html=True)

members = current_members()
with st.sidebar:
    st.markdown("## Thiết lập dự báo")
    symbol = st.selectbox("Cổ phiếu", members, index=members.index("FPT"))
    frame, metadata = load_stock(symbol)
    dates = eligible_cutoffs(symbol)
    latest = st.toggle("Dùng phiên mới nhất trong dữ liệu", value=True)
    cutoff = dates[-1] if latest else st.selectbox("Ngày chốt trong quá khứ", dates[::-1], index=min(25, len(dates) - 1))
    with st.expander("Cài đặt mô hình"):
        st.write(f"Kronos-small · {LOOKBACK} nến đầu vào · {HORIZON} phiên dự báo")
        st.write(f"T={T} · top-p={TOP_P} · {SAMPLE_COUNT} đường mẫu")
        seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
    run = st.button("Chạy dự báo Kronos", type="primary", use_container_width=True)
    source = "Vietcap (VCI)" if metadata.get("latest_source") else "Vietcap (VCI) qua Vnstock"
    st.caption(f"Dữ liệu {symbol} tới **{frame.index[-1]}** · nguồn {source} · truy hồi {metadata.get('retrieved_at_utc', 'không rõ')[:10]}.")
    st.caption("Rổ 30 mã lấy theo kỳ thành viên bắt đầu 03/08/2026 trong bộ nghiên cứu.")

st.write("")
left, right, third = st.columns([1, 1, 1])
left.metric("Mã đang xem", symbol)
right.metric("Ngày chốt đầu vào", cutoff)
third.metric("Phiên giá mới nhất có sẵn", frame.index[-1])

if not run:
    st.markdown('<div class="note">Bấm <b>Chạy dự báo Kronos</b> để thực hiện suy luận thật trên mã và ngày chốt đã chọn. Chọn ngày quá khứ để xem giá thực tế bên cạnh đường dự báo.</div>', unsafe_allow_html=True)
    st.stop()

try:
    prepared = prepare(symbol, cutoff)
    with st.spinner("Kronos đang dự báo trên thiết bị này…"):
        forecast = run_kronos(symbol, cutoff, int(seed))
except Exception as exc:
    st.error(f"Không chạy được dự báo: {exc}")
    st.stop()

last_close = float(prepared.history["close"].iloc[-1])
predicted_close = float(forecast["close"].iloc[-1])
predicted_change = predicted_close / last_close - 1
realized = None
if len(prepared.actual) == HORIZON:
    realized = float(prepared.actual["close"].iloc[-1] / last_close - 1)

st.markdown("### Kết quả cho một mã")
a, b, c, d = st.columns(4)
a.metric("Giá chốt đầu vào", f"{last_close:,.2f}")
b.metric("Giá dự báo phiên 10", f"{predicted_close:,.2f}")
c.metric("Thay đổi dự báo", f"{predicted_change:+.2%}")
d.metric("Thực tế sau 10 phiên" if realized is not None else "Giá tương lai", f"{realized:+.2%}" if realized is not None else "Chưa có")
if realized is not None:
    actual_close = float(prepared.actual["close"].iloc[-1])
    direction_hit = (predicted_change > 0) == (realized > 0)
    st.info(
        f"Đối chiếu lịch sử: Kronos **{'đúng' if direction_hit else 'sai'} hướng**; "
        f"sai số giá đóng cửa phiên 10 là **{abs(predicted_close / actual_close - 1):.2%}**."
    )

components.html(chart_html(prepared.history, forecast, prepared.actual), height=590, scrolling=False)
if prepared.target_dates_estimated:
    st.info("Các ngày dự báo sau ngày dữ liệu cuối được ước bằng ngày làm việc; ngày nghỉ giao dịch thực tế có thể làm lệch nhãn ngày.")
if len(prepared.actual) not in (0, HORIZON):
    st.info(f"Mới có giá thực tế cho {len(prepared.actual)}/{HORIZON} phiên dự báo.")

with st.expander("Xem từng nến dự báo và tải CSV", expanded=True):
    export = forecast[["date", "open", "high", "low", "close"]].copy()
    if not prepared.actual.empty:
        actual_by_date = dict(zip(prepared.actual["date"], prepared.actual["close"]))
        export["actual_close"] = export["date"].map(actual_by_date)
    display = export.copy()
    display.columns = ["Phiên", "Mở dự báo", "Cao dự báo", "Thấp dự báo", "Đóng dự báo"] + (["Đóng thực tế"] if "actual_close" in export else [])
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Tải dự báo CSV", export.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"kronos_{symbol}_{cutoff}_10_phien.csv", mime="text/csv")

st.caption(
    f"Mô hình: NeoQuasar/Kronos-small ({MODEL_REVISION[:8]}) trên CPU. "
    "Dự báo là một đường mẫu, không phải xác suất hay khuyến nghị mua bán. "
    "Giá VCI đã điều chỉnh; đầu vào thiếu `amount`, Kronos ước lượng bằng khối lượng × trung bình OHLC."
)
