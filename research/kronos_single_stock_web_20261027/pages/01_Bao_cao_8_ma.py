"""Precomputed eight-stock report; opening this page never runs Kronos."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parents[1] / "published_reports"
PREFIX = "eight_stocks_20261102"

st.set_page_config(page_title="Kronos · 8 mã · 02/11", page_icon="📊", layout="wide")
st.title("Kronos · 8 mã")
st.caption("Dự báo tính từ giá đóng cửa 02/11/2026. Bản lưu cố định, không tự cập nhật theo giá trong phiên.")

left, right, third = st.columns(3)
for column, suffix, label, mime in (
    (left, "forecasts.csv", "Tải đường dự báo", "text/csv"),
    (right, "manifest.json", "Tải cấu hình và nguồn", "application/json"),
    (third, "fiinx_crosscheck_20261102.json", "Tải đối chiếu giá", "application/json"),
):
    path = ROOT / f"{PREFIX}_{suffix}"
    column.download_button(label, path.read_bytes(), file_name=path.name, mime=mime)

components.html((ROOT / f"{PREFIX}.html").read_text(encoding="utf-8"),
                height=1800, scrolling=True)
