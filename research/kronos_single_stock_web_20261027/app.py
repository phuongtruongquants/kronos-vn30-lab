"""One-stock Kronos viewer with independent paths and a fixed seed audit."""

from __future__ import annotations

import os
import hashlib
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud can rerun app.py in a live interpreter while retaining the
# earlier core module. Reload the local source before importing new symbols.
import core as core_module
importlib.reload(core_module)
from chart import chart_html
from core import HORIZON, LOOKBACK, MODEL_REVISION, T, TOP_P, current_members, eligible_cutoffs, load_stock, prepare, realized_path
from forward import LEDGER, evaluate


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
st.markdown('<div class="subtle">Chọn một cổ phiếu VN30 và xem các đường dự báo 10 phiên từ 90 nến. Kết quả đáng tin cậy phải được lưu trước khi có giá tương lai.</div>', unsafe_allow_html=True)

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
input_bytes = frame.loc[:cutoff].tail(LOOKBACK)[["date", "open", "high", "low", "close", "volume"]].to_csv(index=False).encode()
request_key = (symbol, cutoff, hashlib.sha256(input_bytes).hexdigest(), tuple(seeds), temperature, top_p)
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
    st.markdown('<div class="note">Bấm <b>Chạy dự báo Kronos</b> để tạo các đường mẫu cho mã và ngày chốt đang chọn. Phần kiểm định tiến về sau nằm bên dưới.</div>', unsafe_allow_html=True)
else:
    try:
        prepared = prepare(symbol, cutoff)
        forecasts = saved[1]
        last_close = float(prepared.history["close"].iloc[-1])
        terminal = forecasts.sort_values("date").groupby("seed").tail(1).sort_values("seed")
        median_close = float(terminal["close"].median())
        low_close, high_close = float(terminal["close"].min()), float(terminal["close"].max())
        actual = realized_path(symbol, cutoff)
        realized = float(actual["close"].iloc[-1]) if len(actual) == HORIZON else None
        st.markdown("### Nhiều đường cho một mã")
        a, b = st.columns(2)
        c, d = st.columns(2)
        a.metric("Giá chốt đầu vào", f"{last_close:,.2f}")
        b.metric("Trung vị giá phiên 10", f"{median_close:,.2f}")
        c.metric("Khoảng các mẫu phiên 10", f"{low_close:,.2f}–{high_close:,.2f}")
        d.metric("Đóng cửa thực tế phiên 10" if realized is not None else "Thực tế phiên 10", f"{realized:,.2f}" if realized is not None else "Chưa có")
        st.caption("Các đường là mẫu từ cùng một checkpoint; khoảng thấp–cao không phải khoảng tin cậy hay xác suất đã hiệu chỉnh.")
        components.html(chart_html(prepared.history, forecasts, actual), height=635, scrolling=False)
        st.info("Ngày trên đường dự báo chỉ là ngày làm việc ước tính tại ngày chốt. Khi so sánh, phiên dự báo thứ 10 được ghép với phiên giao dịch thực tế thứ 10; ngày nghỉ có thể khiến hai nhãn ngày khác nhau.")
        if len(actual) not in (0, HORIZON):
            st.info(f"Đã có giá thực tế cho {len(actual)}/{HORIZON} phiên sau ngày chốt.")
        with st.expander("Giá phiên 10 theo seed và tải toàn bộ đường", expanded=True):
            final = terminal[["seed", "date", "close", "volume"]].copy()
            final["thay_đổi"] = final["close"] / last_close - 1
            st.dataframe(final.rename(columns={"seed":"Seed", "date":"Phiên", "close":"Giá dự báo", "volume":"Khối lượng dự báo", "thay_đổi":"Thay đổi"}),
                         use_container_width=True, hide_index=True)
            st.download_button("Tải tất cả đường CSV", forecasts.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"kronos_{symbol}_{cutoff}_{path_count}_duong.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Không hiển thị được dự báo: {exc}")

st.markdown("### Kiểm định tiến về sau")
st.warning("Bảng win rate hồi cứu trước đây đã được gỡ khỏi giao diện: giá lịch sử điều chỉnh được truy hồi vào 26/10/2026, nên không thể xác nhận từng nến đúng phiên bản có sẵn tại ngày chốt. Chọn seed từ 6 cửa sổ rồi kiểm tra 4 cửa sổ cũng quá dễ overfit.")
st.markdown("Lô mới khóa trước kết quả gồm **30 cổ phiếu × 3 seed cố định (42–44) × 10 phiên**. Dự báo, 90 nến đầu vào, cấu hình và dấu vân tay dữ liệu được lưu cùng thời điểm. Chỉ sau khi có đủ 10 phiên thực tế mới tính đúng hướng từng seed; không chọn lại seed theo lô đang kiểm tra.")
runs = sorted(LEDGER.glob("20*")) if LEDGER.exists() else []
if not runs:
    st.info("Chưa có lô dự báo được niêm phong. Vì thế hiện chưa có win rate ngoài mẫu để công bố.")
else:
    for run_dir in runs:
        try:
            import json
            manifest = json.loads((run_dir / "manifest.json").read_text())
            panel = evaluate(run_dir)
            st.caption(f"Lô {manifest['cutoff']} · niêm phong {manifest['sealed_at_ict']} · {len(manifest['symbols'])} mã · seed {manifest['seeds']} · T={manifest['temperature']}, top-p={manifest['top_p']}")
            if panel.empty:
                st.info("Lô này chưa đủ 10 phiên thực tế. Win rate sẽ chỉ xuất hiện sau khi bổ sung dữ liệu tương lai.")
            else:
                complete = panel.symbol.nunique() == len(manifest["symbols"])
                if not complete:
                    st.warning(f"Mới đủ kết quả cho {panel.symbol.nunique()}/{len(manifest['symbols'])} mã; chưa công bố win rate toàn rổ.")
                else:
                    summary = panel.groupby("seed", as_index=False).agg(correct=("hit", "sum"), samples=("hit", "size"))
                    summary["win_rate"] = summary.correct / summary.samples
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                    st.caption("Đây là tỷ lệ đúng hướng giá đóng cửa phiên 10 theo từng cổ phiếu. Các mã cùng giai đoạn có tương quan; 30 mã không phải 30 phép thử độc lập, và chưa bao gồm chi phí giao dịch.")
        except Exception as exc:
            st.error(f"Lô {run_dir.name} không đạt kiểm tra toàn vẹn: {exc}")

st.caption(f"Kronos-small ({MODEL_REVISION[:8]}) · đầu vào 90 nến tới ngày chốt · mốc dự báo ước theo ngày làm việc. Các đường lấy mẫu không phải xác suất hiệu chỉnh. Kiểm định lịch sử với giá truy hồi chỉ dùng để khám phá, không dùng làm bằng chứng lợi thế giao dịch.")
