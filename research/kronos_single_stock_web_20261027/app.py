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
from core import EXTRA_STOCKS, HORIZON, LOOKBACK, MODEL_REVISION, T, TOP_P, current_members, eligible_cutoffs, load_stock, prepare, realized_path
from extra_replay import RUN as DGW_RUN
from forward import LEDGER, evaluate, sha256
from retrospective import RUN as RETRO_RUN


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


def show_stock_replay(symbol: str, stock_summary: pd.DataFrame,
                      stock_panel: pd.DataFrame, stock_outcomes: pd.DataFrame) -> None:
    st.markdown(f"**Riêng {symbol}**")
    summary = stock_summary[["seed", "correct", "samples", "win_rate"]].copy()
    summary["win_rate"] = summary.win_rate.map(lambda value: f"{value:.1%}")
    st.dataframe(summary.rename(columns={"seed": "Seed", "correct": "Đúng hướng",
                                         "samples": "Số mẫu", "win_rate": "Tỷ lệ đúng"}),
                 use_container_width=True, hide_index=True)
    st.markdown(f"**Từng lần dự báo của {symbol}**")
    trials = stock_panel[["cutoff", "seed", "forecast_return", "actual_return", "hit"]].copy()
    trials = trials.merge(stock_outcomes[["cutoff", "forecast_date_10", "actual_date_10"]],
                          on="cutoff", validate="many_to_one")
    trials["forecast_return"] = trials.forecast_return.map(lambda value: f"{value:+.2%}")
    trials["actual_return"] = trials.actual_return.map(lambda value: f"{value:+.2%}")
    trials["hit"] = trials.hit.map({1: "Đúng", 0: "Sai"})
    st.dataframe(trials.rename(columns={"cutoff": "Ngày chốt", "seed": "Seed",
                                        "forecast_return": "Dự báo phiên 10",
                                        "actual_return": "Thực tế phiên 10",
                                        "forecast_date_10": "Ngày dự báo ước tính",
                                        "actual_date_10": "Ngày thực tế", "hit": "Hướng"}),
                 use_container_width=True, hide_index=True)


st.markdown('<div class="eyebrow">Thử riêng mô hình · không có chiến lược giao dịch</div>', unsafe_allow_html=True)
st.title("Kronos Lab")
st.markdown('<div class="subtle">Chọn một trong 30 cổ phiếu VN30 hoặc DGW để xem các đường dự báo 10 phiên từ 90 nến. Kết quả đáng tin cậy phải được lưu trước khi có giá tương lai.</div>', unsafe_allow_html=True)

members = current_members()
with st.sidebar:
    st.markdown("## Thiết lập dự báo")
    symbol = st.selectbox("Cổ phiếu", sorted(members + list(EXTRA_STOCKS)),
                          index=sorted(members + list(EXTRA_STOCKS)).index("FPT"),
                          format_func=lambda ticker: f"{ticker} · mã bổ sung ngoài VN30" if ticker in EXTRA_STOCKS else ticker)
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
    source = ("Vietcap (VCI) trực tiếp" if symbol in EXTRA_STOCKS else
              "Vietcap (VCI)" if metadata.get("latest_source") else "Vietcap (VCI) qua Vnstock")
    st.caption(f"Dữ liệu {symbol} tới **{frame.index[-1]}** · nguồn {source} · truy hồi {metadata.get('retrieved_at_utc','không rõ')[:10]}.")
    st.caption("Rổ 30 mã theo kỳ thành viên bắt đầu 03/08/2026; DGW là mã bổ sung ngoài rổ.")

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
        components.html(chart_html(prepared.history, forecasts, actual), height=710, scrolling=False)
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

st.markdown("### Giả lập hồi cứu trên giá lịch sử")
st.caption("Cố định lịch từ 01/07/2025 đến 27/10/2026, dự báo 10 phiên sau mỗi 10 phiên, ba seed 42–44. Không chọn seed thắng từ kết quả. Chỉ loại cửa sổ thiếu 90 nến đầu vào, có nến đầu vào không hợp lệ hoặc thiếu 10 phiên kết quả. Chấm theo thứ tự phiên; ngày dự báo có thể lệch lịch giao dịch quanh ngày nghỉ.")
st.warning("Đây là **hồi cứu thăm dò**, không phải win rate ngoài mẫu chống leakage hoàn toàn: giá VCI/Vnstock của 30 mã được truy hồi 26/10/2026 và bổ sung 27/10; DGW được lấy 28/10. Rổ 30 mã theo kỳ tháng 8/2026, không phải rổ tại từng mốc cũ. Tải thêm cùng loại giá lịch sử không khôi phục được phiên bản dữ liệu thật sự có ở mỗi ngày quá khứ.")
if not (RETRO_RUN / "manifest.json").exists():
    st.info("Lô hồi cứu dài hơn đang được tính; bảng kết quả sẽ xuất hiện sau khi hoàn tất và kiểm tra đủ đường dự báo.")
else:
    try:
        import json
        retro_manifest = json.loads((RETRO_RUN / "manifest.json").read_text())
        for name, expected_hash in retro_manifest["file_sha256"].items():
            if sha256(RETRO_RUN / f"{name}.csv") != expected_hash:
                raise ValueError(f"Tệp {name}.csv không khớp bản kết quả đã khóa")
        retro_summary = pd.read_csv(RETRO_RUN / "summary.csv")
        retro_stock = pd.read_csv(RETRO_RUN / "per_stock.csv")
        retro_dates = pd.read_csv(RETRO_RUN / "per_date.csv")
        retro_panel = pd.read_csv(RETRO_RUN / "panel.csv")
        retro_outcomes = pd.read_csv(RETRO_RUN / "outcomes.csv")
        retro_exclusions = pd.read_csv(RETRO_RUN / "exclusions.csv")
        st.markdown("**Toàn rổ 30 mã VN30 (không gồm DGW)**")
        a, b, c, d = st.columns(4)
        a.metric("Mốc lịch cố định", len(retro_manifest["anchors"]))
        b.metric("Cặp mã × mốc hợp lệ", retro_manifest["selected_pairs"])
        c.metric("Cặp bị loại và ghi lý do", retro_manifest["excluded_pairs"])
        d.metric("Cặp lệch ngày ước tính", int(retro_outcomes.calendar_mismatch.sum()))
        view = retro_summary[["seed", "correct", "samples", "win_rate", "momentum_rate", "always_up_rate", "mean_abs_return_error"]].copy()
        for field in ("win_rate", "momentum_rate", "always_up_rate", "mean_abs_return_error"):
            view[field] = view[field].map(lambda value: f"{value:.1%}")
        st.dataframe(view.rename(columns={"seed":"Seed", "correct":"Đúng hướng", "samples":"Số mẫu", "win_rate":"Tỷ lệ đúng",
                                          "momentum_rate":"Đà giá 10 phiên", "always_up_rate":"Luôn đoán tăng",
                                          "mean_abs_return_error":"Sai số lợi suất TB"}), use_container_width=True, hide_index=True)
        st.caption("‘Luôn đoán tăng’ dự báo giá phiên 10 cao hơn giá tại mốc ở MỌI cặp mã–mốc; không liên quan đến 5 đường mẫu trên biểu đồ. ‘Đà giá 10 phiên’ dự báo hướng 10 phiên tới giống hướng 10 phiên vừa qua. Cả hai dùng cùng mẫu với Kronos.")
        mismatch_view = retro_panel.groupby(["seed", "calendar_mismatch"], as_index=False).agg(
            correct=("hit", "sum"), samples=("hit", "size"))
        mismatch_view["win_rate"] = (mismatch_view.correct / mismatch_view.samples).map(lambda value: f"{value:.1%}")
        mismatch_view["calendar_mismatch"] = mismatch_view.calendar_mismatch.map({0: "Trùng ngày", 1: "Lệch ngày"})
        with st.expander("Kiểm tra độ nhạy với ngày nghỉ"):
            st.dataframe(mismatch_view.rename(columns={"seed":"Seed", "calendar_mismatch":"Nhãn ngày",
                "correct":"Đúng hướng", "samples":"Số mẫu", "win_rate":"Tỷ lệ đúng"}),
                use_container_width=True, hide_index=True)
            st.caption("Nhóm lệch ngày chỉ là phân tích sau kết quả; không được dùng để chọn mẫu hoặc chọn seed cho tỷ lệ chính.")
        st.caption("Bảng tổng hợp và đồ thị theo ngày bên dưới chỉ gồm 30 mã VN30 trong lô đã khóa; DGW được đánh giá riêng.")
        if symbol in EXTRA_STOCKS:
            if not (DGW_RUN / "manifest.json").exists():
                st.info("DGW đang được tính hồi cứu riêng; dự báo theo mã vẫn dùng được.")
            else:
                dgw_manifest = json.loads((DGW_RUN / "manifest.json").read_text())
                for name, expected_hash in dgw_manifest["file_sha256"].items():
                    if sha256(DGW_RUN / f"{name}.csv") != expected_hash:
                        raise ValueError(f"DGW {name}.csv không khớp bản kết quả đã khóa")
                dgw_summary = pd.read_csv(DGW_RUN / "summary.csv")
                dgw_panel = pd.read_csv(DGW_RUN / "panel.csv")
                dgw_outcomes = pd.read_csv(DGW_RUN / "outcomes.csv")
                st.caption(f"DGW ngoài VN30: {dgw_manifest['selected_windows']}/{len(dgw_manifest['anchors'])} mốc đủ dữ liệu; "
                           f"{int(dgw_outcomes.calendar_mismatch.sum())} mốc lệch ngày ước tính. Dùng cùng seed và lịch như lô VN30.")
                st.warning("Chỉ có 33 mốc DGW và giá lịch sử được truy hồi sau đó. Không chọn seed thắng nhất trên bảng này rồi xem đó là xác suất hay lợi thế giao dịch đã kiểm chứng.")
                show_stock_replay(symbol, dgw_summary, dgw_panel, dgw_outcomes)
                st.download_button("Tải kết quả DGW theo mốc và seed", (DGW_RUN / "panel.csv").read_bytes(),
                                   file_name="kronos_DGW_retrospective_panel.csv", mime="text/csv")
        else:
            show_stock_replay(symbol, retro_stock.loc[retro_stock.symbol.eq(symbol)],
                              retro_panel.loc[retro_panel.symbol.eq(symbol)],
                              retro_outcomes.loc[retro_outcomes.symbol.eq(symbol)])
        timeline = retro_dates.pivot(index="cutoff", columns="seed", values="hit_rate")
        timeline.columns = [f"Seed {seed}" for seed in timeline.columns]
        timeline["Đà giá 10 phiên"] = retro_dates.loc[retro_dates.seed.eq(42)].set_index("cutoff")["momentum_rate"]
        timeline["Luôn đoán tăng"] = retro_dates.loc[retro_dates.seed.eq(42)].set_index("cutoff")["always_up_rate"]
        st.line_chart(timeline, height=260)
        st.caption("Mỗi điểm là tỷ lệ đúng trên các mã đủ dữ liệu tại một mốc; các mã cùng ngày có tương quan nên tổng số cặp không tương đương số quan sát độc lập.")
        st.caption("Lý do loại cửa sổ: " + ", ".join(f"{reason}: {count}"
                   for reason, count in retro_exclusions.reason.value_counts().items()))
        st.download_button("Tải kết quả từng mã × mốc × seed", (RETRO_RUN / "panel.csv").read_bytes(),
                           file_name="kronos_retrospective_panel.csv", mime="text/csv")
        st.download_button("Tải danh sách mốc bị loại", (RETRO_RUN / "exclusions.csv").read_bytes(),
                           file_name="kronos_retrospective_exclusions.csv", mime="text/csv")
    except Exception as exc:
        st.error(f"Không đọc được lô hồi cứu: {exc}")

st.markdown("### Kiểm định tiến về sau")
st.markdown("Lô mới khóa trước kết quả gồm **30 cổ phiếu × 3 seed cố định (42–44) × 10 phiên**. Dự báo, 90 nến đầu vào, cấu hình và dấu vân tay dữ liệu được lưu cùng thời điểm. Chỉ sau khi có đủ 10 phiên thực tế mới tính đúng hướng từng seed; không chọn lại seed theo lô đang kiểm tra.")
if symbol in EXTRA_STOCKS:
    st.caption("DGW chưa nằm trong lô niêm phong tiến về sau của 30 mã VN30 ngày 27/10/2026.")
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
