# Kronos Lab · dự báo một mã VN30

Trang Streamlit chạy **checkpoint Kronos-small thật** cho một mã cổ phiếu ở một ngày chốt, không ghép công thức Codex hay quy tắc phân bổ vốn. Mặc định dùng 90 nến để dự báo 10 phiên, giống phép thử VN30 trước đó. Có thể chọn ngày quá khứ để chồng giá thực tế lên đường dự báo.

```bash
cd '/Users/macbookpro/Test some strategy'
python3 -m streamlit run research/kronos_single_stock_web_20261027/app.py --server.port 8501 --server.fileWatcherType none
```

Mở `http://localhost:8501`. Lần đầu bấm chạy, checkpoint được nạp từ Hugging Face cache và có thể mất thời gian. Nếu checkout Kronos tạm cũ không còn, ứng dụng lấy đúng commit `67b630e...` vào `~/.cache/kronos-single-stock/Kronos`. Có thể đặt `KRONOS_ROOT` trỏ đến checkout cùng commit.

Cần `streamlit`, `torch`, `pandas`, `numpy`, `einops==0.8.1`, `huggingface_hub`, `safetensors`. Mô hình chạy trong tiến trình riêng trên CPU để tránh lỗi của MPS trong phiên web trên máy này.

Dữ liệu lịch sử là bản OHLCV đã điều chỉnh từ VCI qua Vnstock, lưu trong `research/vn30_long_20261026/data/` và kết thúc ngày 23/10/2026. Lớp bổ sung riêng cho trang tại `data_update_20261026/` chứa nến 26/10/2026 của cả 30 mã, lấy trực tiếp từ API biểu đồ Vietcap ngày 27/10/2026. Script `refresh_oct26.py` đã kiểm tra OHLCV phiên 23/10 khớp chính xác với dữ liệu cũ trước khi ghi; dữ liệu backtest cũ không bị sửa. Trang hiển thị ngày dữ liệu mới nhất, **không tự nhận là giá trực tiếp**. Danh sách mã lấy từ kỳ thành viên VN30 bắt đầu 03/08/2026 trong bộ nghiên cứu. Với ngày chốt mới nhất, nhãn 10 ngày tương lai được ước theo ngày làm việc; ngày nghỉ giao dịch có thể khác. Các dự báo lịch sử dùng lịch phiên đã quan sát, chỉ đưa nến đến ngày chốt vào mô hình.

Biểu đồ có thể phóng to hoặc thu nhỏ bằng nút `+`/`−` hay cuộn chuột, kéo ngang để xem phiên khác, và tự chỉnh trục tung theo các nến đang nhìn. Nến lịch sử được tô màu theo chiều tăng giảm; nến Kronos có đường viền đứt; đường đóng cửa Kronos là nét đứt màu xanh lam; đường thực tế sau ngày chốt là màu cam. Nút `Nến` đổi sang chế độ đường giá.

Mô hình cố định: upstream commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, model revision `901c26c1332695a2a8f243eb2f37243a37bea320`, tokenizer revision `0e0117387f39004a9016484a186a908917e22426`, `T=0.6`, `top_p=0.9`, `sample_count=1`. Một đường mẫu không phải phân phối xác suất. Dữ liệu không có `amount`; upstream Kronos suy ra từ `volume * mean(OHLC)`.
