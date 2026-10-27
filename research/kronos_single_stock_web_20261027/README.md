# Kronos Lab · VN30

Ứng dụng Streamlit công khai: https://kronos-vn30-lab.streamlit.app/

Chọn một trong 30 cổ phiếu VN30 và một ngày chốt. Ứng dụng đưa 90 nến OHLCV vào checkpoint `NeoQuasar/Kronos-small` để tạo 10 nến tiếp theo. Chạy từng seed độc lập (`sample_count=1`) để hiển thị 1, 3 hoặc 5 đường giá/khối lượng, bảng giá phiên 10 và CSV toàn bộ dự báo. Biểu đồ có thể phóng to, kéo ngang, tự chỉnh trục giá theo vùng xem và chuyển giữa đường giá/nến.

## Dữ liệu và khả năng tái lập

- Rổ cổ phiếu theo kỳ thành viên VN30 bắt đầu 03/08/2026 trong bộ nghiên cứu. Giá lịch sử VCI đã điều chỉnh, lưu cùng repo. Hai lớp bổ sung `data_update_20261026/` và `data_update_20261027/` chứa phiên 26 và 27/10/2026, lấy từ Vietcap. Từng lớp đối chiếu phiên trùng gần nhất với dữ liệu trước khi thêm. Đây là bản chụp dữ liệu tới 27/10/2026, không phải nguồn giá tự làm mới.
- Mã nguồn Kronos được ghim ở upstream commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, model revision `901c26c1332695a2a8f243eb2f37243a37bea320`, tokenizer revision `0e0117387f39004a9016484a186a908917e22426`. Ứng dụng chạy CPU. Mỗi dự báo nạp mô hình một lần trong tiến trình riêng rồi lấy mẫu tuần tự với các seed cố định.
- Trang đặt mặc định **T=0,6** để thử nghiệm; mặc định hàm `KronosPredictor.predict()` của upstream là **T=1,0**. **Top-p=0,9** là mặc định của upstream. `sample_count>1` trong upstream trả trung bình các mẫu, nên không thể dùng trực tiếp để lấy nhiều đường riêng.
- Dữ liệu không có `amount`; Kronos ước lượng bằng `volume × mean(OHLC)`. Mọi nhãn ngày đích đều ước theo ngày làm việc biết trước tại ngày chốt. Khi đánh giá, bước thứ 10 được ghép với phiên giao dịch thực tế thứ 10, bất kể nhãn ngày ước tính.

## Chống nhìn trước và đánh giá tiến về sau

Đầu vào mô hình chỉ gồm 90 nến đến ngày chốt; nhãn giá thực tế được đọc bằng hàm riêng sau khi dự báo. Tiến trình chạy mô hình không gọi hàm đọc nhãn. Lịch phiên thực tế sau ngày chốt không được đưa vào mô hình. Bảng win rate hồi cứu cũ đã được gỡ khỏi giao diện: dữ liệu giá từ 2018 được truy hồi năm 2026 dưới dạng điều chỉnh, nên không có bằng chứng phiên bản giá có sẵn tại từng mốc. Danh sách 30 thành viên hiện tại cũng không thể dùng như rổ lịch sử.

Lệnh `python3 research/kronos_single_stock_web_20261027/forward.py freeze` niêm phong một lô gồm 30 mã × 3 seed cố định × 10 phiên sau khi đủ giá phiên hôm nay. Lô chứa nến đầu vào, đường dự báo, thời điểm lưu, mã băm SHA-256, mã băm nguồn chạy, phiên bản model/tokenizer và môi trường tính. Lệnh từ chối ghi đè hoặc chạy trên dữ liệu cũ. Lô cần được commit/push công khai trước phiên tiếp theo để có dấu thời gian kiểm chứng độc lập. `forward.py evaluate` chỉ chấm mã đã đủ 10 phiên thực tế; trên giao diện, win rate toàn rổ chỉ hiển thị khi đủ cả 30 mã. Nếu lịch giao dịch thực tế lệch lịch ngày làm việc mà Kronos đã nhận, hoặc giá ngày chốt bị điều chỉnh lại, lô bị chặn chấm và cần xử lý công khai trước. Không chọn lại seed/T/top-p sau khi xem kết quả. Những lần bấm dự báo tùy ý trên giao diện không được tính vào win rate này.

Checkpoint model/tokenizer đã có tệp trọng số công bố ngày 30/06/2025, trước các phiên 2026 trong lô tiến về sau; tuy nhiên ngày cuối cùng của dữ liệu huấn luyện chưa được nhà phát triển xác nhận. Kết quả trên nhiều cổ phiếu cùng 10 phiên có tương quan và không tương đương 30 quan sát độc lập; win rate đúng hướng cũng không bao gồm chi phí hay bảo đảm lợi nhuận. Seed chỉ cố định phép lấy mẫu trong cùng môi trường; khác hệ điều hành/PyTorch có thể cho đường khác.

## Chạy trên máy

Từ gốc repo:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run research/kronos_single_stock_web_20261027/app.py
```
