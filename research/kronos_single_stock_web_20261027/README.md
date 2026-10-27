# Kronos Lab · VN30

Ứng dụng Streamlit công khai: https://kronos-vn30-lab.streamlit.app/

Chọn một trong 30 cổ phiếu VN30 và một ngày chốt. Ứng dụng đưa 90 nến OHLCV vào checkpoint `NeoQuasar/Kronos-small` để tạo 10 nến tiếp theo. Chạy từng seed độc lập (`sample_count=1`) để hiển thị 1, 3 hoặc 5 đường giá/khối lượng, bảng giá phiên 10 và CSV toàn bộ dự báo. Biểu đồ có thể phóng to, kéo ngang, tự chỉnh trục giá theo vùng xem và chuyển giữa đường giá/nến.

## Dữ liệu và khả năng tái lập

- Rổ cổ phiếu theo kỳ thành viên VN30 bắt đầu 03/08/2026 trong bộ nghiên cứu. Giá lịch sử VCI đã điều chỉnh, lưu cùng repo. Hai lớp bổ sung `data_update_20261026/` và `data_update_20261027/` chứa phiên 26 và 27/10/2026, lấy từ Vietcap. Từng lớp đối chiếu phiên trùng gần nhất với dữ liệu trước khi thêm. Đây là bản chụp dữ liệu tới 27/10/2026, không phải nguồn giá tự làm mới.
- Mã nguồn Kronos được ghim ở upstream commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, model revision `901c26c1332695a2a8f243eb2f37243a37bea320`, tokenizer revision `0e0117387f39004a9016484a186a908917e22426`. Ứng dụng chạy CPU. Mỗi dự báo nạp mô hình một lần trong tiến trình riêng rồi lấy mẫu tuần tự với các seed cố định.
- Trang đặt mặc định **T=0,6** để thử nghiệm; mặc định hàm `KronosPredictor.predict()` của upstream là **T=1,0**. **Top-p=0,9** là mặc định của upstream. `sample_count>1` trong upstream trả trung bình các mẫu, nên không thể dùng trực tiếp để lấy nhiều đường riêng.
- Dữ liệu không có `amount`; Kronos ước lượng bằng `volume × mean(OHLC)`. Khi dự báo sau ngày dữ liệu cuối, nhãn ngày tương lai chỉ ước theo ngày làm việc, chưa loại trừ ngày nghỉ giao dịch.

## Kiểm tra seed trên lịch sử

Nút **Tính win rate lịch sử cho mã này** kiểm tra ba seed 42/43/44 trên mười cửa sổ 10 phiên không chồng lấn, ấn định từ lịch giao dịch 2026. Sáu cửa sổ trước dùng để chọn seed theo tỷ lệ dự báo đúng hướng giá đóng cửa phiên 10, rồi theo sai số giá nếu hòa. Bốn cửa sổ sau để báo kết quả kiểm tra. Giao diện hiển thị tỷ lệ riêng từng seed, seed được chọn và CSV chi tiết từng mốc. Dữ liệu tương lai không được đưa vào đầu vào tại mốc dự báo.

Kiểm tra này chỉ hạn chế nhìn trước trong đầu vào và thiên lệch do chọn seed. Giá điều chỉnh có thể được sửa lại sau sự kiện; chưa xác minh được ngày kết thúc tập huấn luyện của checkpoint. Không coi tỷ lệ này là bằng chứng hoàn toàn ngoài mẫu hay hiệu quả giao dịch sau chi phí.

Seed chỉ cố định phép lấy mẫu trong một môi trường chạy. Số dự báo có thể khác giữa máy Mac và Linux hay giữa các phiên bản PyTorch; so sánh win rate cần giữ nguyên môi trường, checkpoint và bản dữ liệu.

## Chạy trên máy

Từ gốc repo:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run research/kronos_single_stock_web_20261027/app.py
```
