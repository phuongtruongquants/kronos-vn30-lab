# Kronos Lab · VN30

Ứng dụng Streamlit công khai: https://kronos-vn30-lab.streamlit.app/

Chọn một trong 30 cổ phiếu VN30 hoặc **DGW (mã bổ sung ngoài rổ)** và một ngày chốt. Ứng dụng đưa 90 nến OHLCV vào checkpoint `NeoQuasar/Kronos-small` để tạo 10 nến tiếp theo. Chạy từng seed độc lập (`sample_count=1`) để hiển thị 1, 3 hoặc 5 đường giá/khối lượng, bảng giá phiên 10 và CSV toàn bộ dự báo. Biểu đồ có thể phóng to, kéo ngang, tự chỉnh trục giá theo vùng xem và chuyển giữa đường giá/nến. Chế độ nến vẽ OHLC dự báo của một seed được chọn; các seed còn lại là đường nét đứt. Nến OHLC dự báo bất nhất được đánh dấu `!`, thân/bóng chỉ được bao trùm để hiển thị; CSV và tooltip giữ nguyên giá trị gốc.

## Dữ liệu và khả năng tái lập

- Rổ cổ phiếu theo kỳ thành viên VN30 bắt đầu 03/08/2026 trong bộ nghiên cứu. Giá lịch sử VCI đã điều chỉnh, lưu cùng repo. Các lớp `data_update_20261026/`, `data_update_20261027/` và `data_update_20261028/` bổ sung từng phiên; lớp 28/10 có 30 mã VN30 cùng DGW. Mỗi nến mới được đối chiếu phiên trùng 27/10 với nguồn Vietcap và khớp OHLCV. Đây là bản chụp dữ liệu tới 28/10/2026, không phải nguồn giá tự làm mới.
- Mã nguồn Kronos được ghim ở upstream commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a`, model revision `901c26c1332695a2a8f243eb2f37243a37bea320`, tokenizer revision `0e0117387f39004a9016484a186a908917e22426`. Ứng dụng chạy CPU. Mỗi dự báo nạp mô hình một lần trong tiến trình riêng rồi lấy mẫu tuần tự với các seed cố định.
- Trang đặt mặc định **T=1,0** và **top-p=0,9**, khớp mặc định hàm `KronosPredictor.predict()` của upstream. `sample_count>1` trong upstream trả trung bình các mẫu, nên mỗi đường được chạy riêng với `sample_count=1` và một seed. Số đường cho kết quả tăng là tần suất lấy mẫu, không phải xác suất đã hiệu chỉnh.
- Dữ liệu không có `amount`; Kronos ước lượng bằng `volume × mean(OHLC)`. Mọi nhãn ngày đích đều ước theo ngày làm việc biết trước tại ngày chốt. Khi đánh giá, bước thứ 10 được ghép với phiên giao dịch thực tế thứ 10, bất kể nhãn ngày ước tính.
- DGW lấy trực tiếp từ cùng API biểu đồ Vietcap (VCI) cho chuỗi gốc 03/08/2015–27/10/2026 trong `extra_stock_data/`; phiên 28/10 nằm trong lớp bổ sung riêng. Sáu nến lịch sử có khối lượng 0 được giữ nguyên; cửa sổ 90 nến chứa chúng không đủ điều kiện chạy mô hình. Nguồn là bản chụp truy hồi, không phải phiên bản giá point-in-time.

## Chống nhìn trước và đánh giá tiến về sau

Đầu vào mô hình chỉ gồm 90 nến đến ngày chốt; nhãn giá thực tế được đọc bằng hàm riêng sau khi dự báo. Tiến trình chạy mô hình không gọi hàm đọc nhãn. Lịch phiên thực tế sau ngày chốt không được đưa vào mô hình. Bảng nhỏ 10 mốc/4 mốc kiểm tra trước đây đã được thay bằng lô hồi cứu dài hơn, gắn nhãn thăm dò: dữ liệu giá từ 2018 được truy hồi năm 2026 dưới dạng điều chỉnh, nên không có bằng chứng phiên bản giá có sẵn tại từng mốc. Danh sách 30 thành viên hiện tại cũng không thể dùng như rổ lịch sử.

## Giả lập hồi cứu để khám phá

`retrospective.py` dùng chính checkpoint Kronos đã ghim để dự báo từng mã của rổ tháng 8/2026 tại các mốc cố định từ 01/07/2025 đến 27/10/2026, cách nhau 10 phiên. Ba seed 42–44, lookback 90 và horizon 10 được cố định trước; không chọn seed tốt nhất sau khi nhìn kết quả. Chỉ loại cửa sổ thiếu 90 nến đầu vào hoặc 10 phiên giá kết quả; mọi cửa sổ bị loại được lưu cùng lý do. Dự báo ghi ngày làm việc ước tính tại mốc chốt, nhưng chấm theo phiên giao dịch thực tế thứ 10; số cặp lệch nhãn ngày được công bố, không dùng để lọc mẫu. Dự báo được tính và ghi ra tệp trước khi mã ghép giá thực tế; bảng so sánh thêm quy tắc theo đà giá 10 phiên và luôn đoán tăng trên đúng cùng mẫu.

Lô T=0,6 được giữ nguyên. Lô T=1,0 chạy lại trên **đúng cùng** 956 cặp mã–mốc, 90 nến đầu vào, kết quả thực tế, checkpoint, seed và top-p; chương trình kiểm tra mã băm đầu vào và kết quả so với lô T=0,6 trước khi công bố. DGW được chạy và đối chiếu riêng trên cùng 33 mốc cũ. Bảng tần suất tăng thực tế theo số đường mẫu dự báo tăng chỉ là thống kê hồi cứu của ba seed; giao diện có thể vẽ năm seed nên tỷ lệ 1/5–5/5 trên một lần bấm không được diễn giải thành xác suất đã hiệu chỉnh. T=1,0 được chọn sau khi đã quan sát lô T=0,6, vì vậy chỉ lô dự báo niêm phong từ 28/10 trở đi mới giúp đánh giá thay đổi này tiến về sau.

“Luôn đoán tăng” dự báo `close(t+10) > close(t)` cho **mọi** cặp mã–mốc, độc lập với số đường Kronos đang vẽ. “Đà giá 10 phiên” lấy dấu của `close(t)/close(t-10)-1` làm hướng dự báo cho `close(t+10)/close(t)-1`. Hai quy tắc chỉ làm mốc so sánh, không phải tín hiệu giao dịch đã kiểm chứng.

`extra_replay.py` tính DGW riêng trên cùng 33 mốc, ba seed và horizon. Tệp `extra_replay_runs/` có đầu vào, nến dự báo, giá thực tế, bảng từng mốc và mã băm. DGW không nhập vào mẫu 956 cặp VN30 hoặc lô niêm phong 30 mã. Vì chỉ có 33 mốc DGW, không được chọn seed đúng nhiều nhất sau khi xem kết quả rồi gọi đó là lợi thế hay xác suất đã hiệu chỉnh.

Bộ CSV có nến từ 2018, nhưng không dùng giai đoạn 2018–06/2025 để chấm Kronos: các nến đó có thể đã nằm trong tập huấn luyện. Tệp trọng số checkpoint được [công bố 30/06/2025](https://huggingface.co/NeoQuasar/Kronos-small/commit/ac5c409a313c4eadd1ca78201322f5cadb9c34ab); việc bắt đầu từ 01/07/2025 chỉ loại được nguy cơ mô hình đã học chính các phiên sau ngày phát hành, không sửa được dữ liệu giá truy hồi hay thiên lệch chọn mã.

Đây là **giả lập trên giá lịch sử truy hồi**, không phải kiểm định point-in-time: giá VCI/Vnstock đã điều chỉnh được lấy vào 26/10/2026, và danh sách 30 mã cố định theo rổ tháng 8/2026. Tải lại cùng loại giá lịch sử từ Vnstock hoặc FiinX không tự tạo phiên bản giá từng ngày chốt; cần dữ liệu as-of hoặc giá thô cùng lịch corporate action đáng tin cậy để xử lý nguồn nhìn trước đó. Nhiều mã trong cùng một mốc cùng chịu thị trường nên tổng số cặp mã–mốc không phải số quan sát độc lập.

Lệnh `python3 research/kronos_single_stock_web_20261027/forward.py freeze` niêm phong một lô gồm 30 mã × 3 seed cố định × 10 phiên sau khi đủ giá phiên hôm nay. Lô chứa nến đầu vào, đường dự báo, thời điểm lưu, mã băm SHA-256, mã băm nguồn chạy, phiên bản model/tokenizer và môi trường tính. Lệnh từ chối ghi đè hoặc chạy trên dữ liệu cũ. Lô cần được commit/push công khai trước phiên tiếp theo để có dấu thời gian kiểm chứng độc lập. `forward.py evaluate` chỉ chấm mã đã đủ 10 phiên thực tế; trên giao diện, win rate toàn rổ chỉ hiển thị khi đủ cả 30 mã. Nếu lịch giao dịch thực tế lệch lịch ngày làm việc mà Kronos đã nhận, hoặc giá ngày chốt bị điều chỉnh lại, lô bị chặn chấm và cần xử lý công khai trước. Không chọn lại seed/T/top-p sau khi xem kết quả. Những lần bấm dự báo tùy ý trên giao diện không được tính vào win rate này.

Lô niêm phong 27/10 dùng T=0,6; lô 28/10 dùng T=1,0, cùng 30 mã và ba seed. Chúng là hai tập dự báo tiến về sau ở hai ngày chốt khác nhau, nên chưa thể dùng chúng để kết luận T nào tốt hơn. DGW hiện có hồi cứu riêng nhưng chưa có lô niêm phong tiến về sau.

Checkpoint model/tokenizer đã có tệp trọng số công bố ngày 30/06/2025, trước các phiên 2026 trong lô tiến về sau; tuy nhiên ngày cuối cùng của dữ liệu huấn luyện chưa được nhà phát triển xác nhận. Kết quả trên nhiều cổ phiếu cùng 10 phiên có tương quan và không tương đương 30 quan sát độc lập; win rate đúng hướng cũng không bao gồm chi phí hay bảo đảm lợi nhuận. Seed chỉ cố định phép lấy mẫu trong cùng môi trường; khác hệ điều hành/PyTorch có thể cho đường khác.

## Chạy trên máy

Từ gốc repo:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run research/kronos_single_stock_web_20261027/app.py
```
