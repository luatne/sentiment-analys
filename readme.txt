TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT (Vietnamese Sentiment Analysis Assistant)

Dự án triển khai một ứng dụng web đơn giản sử dụng **Streamlit** và mô hình **DistilBERT-base-multilingual-cased** đã được tinh chỉnh (fine-tuned) để phân loại cảm xúc (Tích cực, Tiêu cực, Trung lập) từ văn bản tiếng Việt.

Tính năng nổi bật

Phân loại cảm xúc:Dự đoán cảm xúc của văn bản đầu vào.
Tiền xử lý nâng cao:Sử dụng `underthesea` để tách từ và xử lý các từ viết tắt, tiếng lóng (`teencode.txt`).
Lưu trữ lịch sử:Lưu trữ và hiển thị lịch sử các lần phân tích vào cơ sở dữ liệu SQLite (`sentiment.db`).
Triển khai trên CPU:Tối ưu để chạy ổn định và hiệu quả trên môi trường CPU tiêu chuẩn.
Công nghệ sử dụng

| Công nghệ | Phiên bản | Mục đích |
| Streamlit | `1.51.0` | Xây dựng giao diện ứng dụng web. |
| PyTorch| `2.9.1` | Khung Deep Learning chính. |
| Hugging Face/Transformers**| `4.57.3` | Tải và chạy mô hình DistilBERT. |
| Underthesea | `8.3.0` | Hỗ trợ tách từ tiếng Việt. |
| SQLite3 | Tích hợp | Lưu trữ lịch sử phân loại. |
Hướng dẫn cài đặt và vận hành

1. Chuẩn bị môi trường

Đảm bảo bạn đã cài đặt Python 3.x. Tạo và kích hoạt môi trường ảo:

```bash
python -m venv venv
source venv/bin/activate  # Trên Linux/macOS
# hoặc
.\venv\Scripts\activate  # Trên Windows	

2.Sử dụng file requirements.txt để cài đặt tất cả các thư viện cần thiết:
pip install -r requirements.txt

3.Chuẩn bị Mô hình (Checkpoint)
Tải file trọng số (checkpoint) link trong github
4. Khởi chạy Ứng dụng
Chạy lệnh Streamlit để khởi động ứng dụng web:
streamlit run app.py