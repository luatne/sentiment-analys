🇻🇳 TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT

Vietnamese Sentiment Analysis Assistant

Dự án này giới thiệu một ứng dụng web gọn nhẹ được xây dựng trên nền tảng Streamlit nhằm mục đích thực hiện nhiệm vụ phân loại cảm xúc (Sentiment Analysis) cho văn bản tiếng Việt. Mô hình cốt lõi được triển khai là DistilBERT-base-multilingual-cased, đã được tinh chỉnh (fine-tuned) trên tập dữ liệu đa dạng để đạt được độ chính xác cao trong việc nhận diện ba nhãn cảm xúc: Tích cực (POSITIVE), Tiêu cực (NEGATIVE), và Trung lập (NEUTRAL).

I. 🌟 Tính năng và Chức năng

Phân loại Cảm xúc: Cung cấp khả năng dự đoán cảm xúc tức thì cho mọi đoạn văn bản tiếng Việt do người dùng nhập vào.

Tiền xử lý Nâng cao (Preprocessing Pipeline): Đảm bảo chất lượng đầu vào của mô hình bằng cách thực hiện các bước:

Chuẩn hóa văn bản về chữ thường (Lowercase).

Xử lý và thay thế các từ viết tắt, tiếng lóng (teencode) dựa trên từ điển tùy chỉnh (teencode.txt).

Tách từ tiếng Việt chuyên sâu (Word Tokenization) sử dụng thư viện underthesea, giúp mô hình hiểu ngữ cảnh tốt hơn.

Quản lý Dữ liệu: Hỗ trợ lưu trữ và hiển thị lịch sử các lần phân tích vào cơ sở dữ liệu SQLite (sentiment.db).

Tối ưu Hiệu suất: Mô hình được cấu hình để chạy ổn định và hiệu quả trên môi trường CPU tiêu chuẩn (CPU-optimized deployment).

II. 🛠️ Công nghệ & Thư viện Chính

Công nghệ

Mục đích

Streamlit

Xây dựng giao diện ứng dụng web tương tác (Front-end/UI).

PyTorch

Khung Deep Learning nền tảng để chạy và quản lý mô hình.

Hugging Face/Transformers

Cung cấp mô hình nền tảng DistilBERT và các công cụ Tokenizer.

Underthesea

Hỗ trợ tách từ tiếng Việt và xử lý ngôn ngữ tự nhiên.

SQLite3

Hệ quản trị cơ sở dữ liệu để lưu trữ lịch sử phân loại.

III. 🚀 Hướng dẫn Cài đặt và Vận hành

1. Chuẩn bị Môi trường Python

Đảm bảo hệ thống của bạn đã cài đặt Python 3.8 trở lên. Khuyến nghị sử dụng môi trường ảo (Virtual Environment) để cô lập các thư viện của dự án:

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Linux/macOS
source venv/bin/activate
# Trên Windows
.\venv\Scripts\activate


2. Cài đặt các Thư viện Phụ thuộc

Sử dụng lệnh pip để cài đặt tất cả các thư viện cần thiết. Nếu bạn có file requirements.txt, hãy sử dụng nó; nếu không, cài đặt trực tiếp:

# Cài đặt trực tiếp các thư viện
pip install streamlit torch transformers underthesea
# Cài đặt từ file requirements.txt (nếu có)
# pip install -r requirements.txt


3. Khởi chạy Ứng dụng

Sau khi cài đặt xong, bạn cần đảm bảo tất cả các file mã nguồn và tài nguyên (app.py, sentiment1.py, preprocess.py, database.py, teencode.txt, và file trọng số mô hình checkpoint5.pth) nằm trong cùng một thư mục.

Chạy ứng dụng bằng lệnh Streamlit:

streamlit run app.py


Ứng dụng sẽ tự động mở trên trình duyệt web tại địa chỉ: http://localhost:8501
