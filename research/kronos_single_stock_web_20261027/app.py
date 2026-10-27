"""One-stock Kronos viewer with independent paths and a fixed seed audit."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from audit import AUDIT_SEEDS, AUDIT_START, CALIBRATION_WINDOWS, fixed_cutoffs, score
from chart import chart_html
from core import HORIZON, LOOKBACK, MODEL_REVISION, T, TOP_P, current_members, eligible_cutoffs, load_stock, prepare


st.set_page_config(page_title="Kronos Lab · VN30", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp{background:#f5f7f6;color:#15231f}.block-container{max-width:1420px;padding-top:2.2rem}
h1,h2,h3{color:#12362d;letter-spacing:-.035em}h1{font-size:clamp(2.5rem,5vw,4.4rem);line-height:1}
[data-testid="stMetric"]{background:#fff;border:1px solid #dce5df;border-radius:16px;padding:16px}
[data-testid="stSidebar"]{background:#e9f0eb}[data-testid="stBaseButton-primary"]{background:#0b8064;border-color:#0b8064}
.eyebrow{color:#17735a;text-transform:uppercase;font-size:.78rem;font-weight:750;letter-spacing:.16em}
.subtle{color:#587067;font-size:1rem}.note{background:#e7f2ed;border-left:4px solid #168365;padding:12px 16px;border-radius:8px;color:#254b3f}
</style>
""", unsafe_allow_html=True)


def run_kronos(symbol: str, cutoffs: list[str], seeds: list[int], temperature: float,
               top_p: float, timeout: int = 300) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="kronos_single_stock_") as directory:
        output = Path(directory) / "forecast.csv"
        env = os.environ.copy()
        env["KRONOS_DEVICE"] = "cpu"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "worker.py"),
             "--symbol", symbol, "--cutoffs", ",".join(cutoffs),
             "--seeds", ",".join(map(str, seeds)),
             "--temperature", str(temperature), "--top-p", str(top_p),
             "--output", str(output)],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        if completed.returncode != 0 or not output.exists():
            detail = completed.stderr[-1600:] or completed.stdout[-1600:]
            raise RuntimeError(detail or f"Kronos worker exited {completed.returncode}")
        return pd.read_csv(output)


st.markdown('<div class="eyebrow">Thử riêng mô hình · không có chiến lược giao dịch</div>', unsafe_allow_html=True)
st.title("Kronos Lab")
st.markdown('<div class="subtle">Chọn một cổ phiếu VN30, xem nhiều đường dự báo 10 phiên từ 90 nến và kiểm tra seed trên lịch sử.</div>', unsafe_allow_html=True)

members = current_members()
with st.sidebar:
    st.markdown("## Thiết lập dự báo")
    symbol = st.selectbox("Cổ phiếu", members, index=members.index("FPT"))
    frame, metadata = load_stock(symbol)
    dates = eligible_cutoffs(symbol)
    latest = st.toggle("Dùng phiên mới nhất trong dữ liệu", value=True)
    cutoff = dates[-1] if latest else st.selectbox("Ngày chốt trong quá khứ", dates[::-1], index=min(25, len(dates)-1))
    path_count = st.select_slider("Số đường dự báo độc lập", options=[1, 3, 5], value=5)
    base_seed = int(st.number_input("Seed đầu tiên", min_value=0, max_value=2**31-6, value=42, step=1))
    with st.expander("Cài đặt lấy mẫu và mô hình"):
        st.write(f"Kronos-small · {LOOKBACK} nến đầu vào · {HORIZON} phiên dự báo")
        temperature = float(st.slider("Temperature (T)", min_value=.2, max_value=1.5, value=float(T), step=.1))
        top_p = float(st.slider("Top-p", min_value=.5, max_value=1.0, value=float(TOP_P), step=.05))
        st.caption("Mỗi seed được chạy riêng với sample_count=1. `sample_count>1` trong Kronos sẽ lấy trung bình, không trả từng đường.")
        st.caption("T=0,6 là cấu hình demo; mặc định `predict()` của Kronos là T=1,0. Top-p=0,9 là mặc định của `predict()`.")
    run = st.button("Chạy dự báo Kronos", type="primary", use_container_width=True)
    source = "Vietcap (VCI)" if metadata.get("latest_source") else "Vietcap (VCI) qua Vnstock"
    st.caption(f"Dữ liệu {symbol} tới **{frame.index[-1]}** · nguồn {source} · truy hồi {metadata.get('retrieved_at_utc','không rõ')[:10]}.")
    st.caption("Rổ 30 mã theo kỳ thành viên bắt đầu 03/08/2026 trong bộ nghiên cứu.")

st.write("")
left, right, third = st.columns(3)
left.metric("Mã đang xem", symbol)
right.metric("Ngày chốt đầu vào", cutoff)
third.metric("Phiên giá mới nhất có sẵn", frame.index[-1])

seeds = list(range(base_seed, base_seed + path_count))
request_key = (symbol, cutoff, tuple(seeds), temperature, top_p)
if run:
    try:
        with st.spinner(f"Kronos đang tạo {path_count} đường độc lập trên CPU…"):
            result = run_kronos(symbol, [cutoff], seeds, temperature, top_p)
        st.session_state["forecast"] = (request_key, result)
    except Exception as exc:
        st.error(f"Không chạy được dự báo: {exc}")
        st.stop()

saved = st.session_state.get("forecast")
if saved is None or saved[0] != request_key:
    st.markdown('<div class="note">Bấm <b>Chạy dự báo Kronos</b> để tạo các đường mẫu cho mã và ngày chốt đang chọn. Phần đánh giá seed lịch sử nằm bên dưới.</div>', unsafe_allow_html=True)
else:
    try:
        prepared = prepare(symbol, cutoff)
        forecasts = saved[1]
        last_close = float(prepared.history["close"].iloc[-1])
        terminal = forecasts.sort_values("date").groupby("seed").tail(1).sort_values("seed")
        median_close = float(terminal["close"].median())
        low_close, high_close = float(terminal["close"].min()), float(terminal["close"].max())
        realized = float(prepared.actual["close"].iloc[-1]) if len(prepared.actual) == HORIZON else None
        st.markdown("### Nhiều đường cho một mã")
        a, b, c, d = st.columns(4)
        a.metric("Giá chốt đầu vào", f"{last_close:,.2f}")
        b.metric("Trung vị giá phiên 10", f"{median_close:,.2f}")
        c.metric("Khoảng các mẫu phiên 10", f"{low_close:,.2f}–{high_close:,.2f}")
        d.metric("Đóng cửa thực tế phiên 10" if realized is not None else "Thực tế phiên 10", f"{realized:,.2f}" if realized is not None else "Chưa có")
        st.caption("Các đường là mẫu từ cùng một checkpoint; khoảng thấp–cao không phải khoảng tin cậy hay xác suất đã hiệu chỉnh.")
        components.html(chart_html(prepared.history, forecasts, prepared.actual), height=635, scrolling=False)
        if prepared.target_dates_estimated:
            st.info("Ngày dự báo sau phiên cuối được ước bằng ngày làm việc; ngày nghỉ giao dịch có thể làm lệch nhãn.")
        if len(prepared.actual) not in (0, HORIZON):
            st.info(f"Đã có giá thực tế cho {len(prepared.actual)}/{HORIZON} phiên dự báo.")
        with st.expander("Giá phiên 10 theo seed và tải toàn bộ đường", expanded=True):
            final = terminal[["seed", "date", "close", "volume"]].copy()
            final["thay_đổi"] = final["close"] / last_close - 1
            st.dataframe(final.rename(columns={"seed":"Seed", "date":"Phiên", "close":"Giá dự báo", "volume":"Khối lượng dự báo", "thay_đổi":"Thay đổi"}),
                         use_container_width=True, hide_index=True)
            st.download_button("Tải tất cả đường CSV", forecasts.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"kronos_{symbol}_{cutoff}_{path_count}_duong.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Không hiển thị được dự báo: {exc}")

st.markdown("### Kiểm tra win rate theo seed")
st.markdown(
    "Lịch kiểm tra được ấn định trước: mỗi 10 phiên từ đầu 2026, lấy 10 cửa sổ gần nhất có đủ giá thực tế. "
    "Ba seed cố định **42, 43, 44**; **6 cửa sổ đầu** để chọn seed, **4 cửa sổ sau** để kiểm tra. "
    "Mỗi cửa sổ dự báo 10 phiên, không dùng nến tương lai làm đầu vào. "
    "Nếu số lần đúng bằng nhau, chọn seed có sai số giá thấp hơn trên 6 cửa sổ đầu."
)
st.caption("Win rate = số lần dự báo đúng dấu thay đổi giá đóng cửa phiên 10 / số cửa sổ. "
           "Seed chỉ thay đổi phép lấy mẫu, không phải trọng số mô hình. Không chọn lại seed hoặc T/top-p sau khi xem cửa sổ kiểm tra.")
audit_dates = fixed_cutoffs(symbol)
st.caption("Mốc lịch sử áp dụng cho mã đang chọn: " + (", ".join(audit_dates) if audit_dates else "chưa đủ dữ liệu"))
audit_key = (symbol, tuple(audit_dates), AUDIT_SEEDS, T, TOP_P, frame.index[-1])
audit_run = st.button("Tính win rate lịch sử cho mã này", disabled=len(audit_dates) < 4)
if audit_run:
    try:
        with st.spinner(f"Đang kiểm tra {len(audit_dates)} mốc × {len(AUDIT_SEEDS)} seed; có thể mất vài phút…"):
            audit_paths = run_kronos(symbol, audit_dates, list(AUDIT_SEEDS), T, TOP_P, timeout=900)
            panel, summary, chosen_seed = score(symbol, audit_paths, audit_dates)
        st.session_state["seed_audit"] = (audit_key, panel, summary, chosen_seed)
    except Exception as exc:
        st.error(f"Không tính được win rate: {exc}")

audit_saved = st.session_state.get("seed_audit")
if audit_saved is not None and audit_saved[0] == audit_key:
    _, panel, summary, chosen_seed = audit_saved
    view = summary.copy()
    view["win_rate"] = view.apply(lambda r: f"{int(r.wins)}/{int(r.windows)} ({r.win_rate:.0%})", axis=1)
    view["avg_abs_price_error"] = view["avg_abs_price_error"].map(lambda x: f"{x:.2%}")
    st.dataframe(view.rename(columns={"seed":"Seed", "phase":"Tập", "wins":"Đúng hướng", "windows":"Số cửa sổ", "win_rate":"Win rate", "avg_abs_price_error":"Sai số giá TB"}),
                 use_container_width=True, hide_index=True)
    if chosen_seed is not None:
        held_out = summary.loc[summary.seed.eq(chosen_seed) & summary.phase.eq("Kiểm tra")].iloc[0]
        st.info(f"Seed được chọn chỉ từ {CALIBRATION_WINDOWS} cửa sổ đầu: **{chosen_seed}**. "
                f"Trên cửa sổ kiểm tra sau đó: **{int(held_out.wins)}/{int(held_out.windows)}** đúng hướng "
                f"({held_out.win_rate:.0%}). Mẫu nhỏ; không suy ra lợi thế giao dịch.")
    st.download_button("Tải chi tiết từng mốc × seed", panel.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"kronos_seed_audit_{symbol}.csv", mime="text/csv")

st.caption(
    f"Mô hình: NeoQuasar/Kronos-small ({MODEL_REVISION[:8]}) trên CPU. "
    "Giá VCI đã điều chỉnh; khi đầu vào thiếu `amount`, Kronos ước lượng bằng khối lượng × trung bình OHLC. "
    "Lịch sử giá điều chỉnh và thời điểm dữ liệu gốc có thể được sửa sau sự kiện; mốc kết thúc huấn luyện checkpoint chưa được xác minh. "
    "Vì vậy bảng trên chỉ kiểm soát nhìn trước trong đầu vào và chọn seed, chưa chứng minh hoàn toàn ngoài mẫu."
)
