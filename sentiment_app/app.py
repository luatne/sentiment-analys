import streamlit as st
# ----------------------------------------------------
# THAY ĐỔI: Import từ sentiment_manual thay vì sentiment
# ----------------------------------------------------
from sentiment1 import process_and_store # Hàm chính thực hiện TOÀN BỘ quy trình
from database import init_db, load_history

# ... (Các phần import và khởi tạo phía trên - giữ nguyên)

# 2. Khởi tạo DB (cần tệp database.py)
init_db()

# 3. Cache pipeline cho Streamlit
try:
    from sentiment1 import _load_model_cached as load_model_cached
except ImportError:
    def load_model_cached():
        return None 

@st.cache_resource
def load_model():
    print("Streamlit caching model from sentiment_manual...")
    return load_model_cached()


# BẮT ĐẦU ỨNG DỤNG STREAMLIT
st.title("Sentiment Analysis App (Manual Load)")
st.write("Phân loại cảm xúc tiếng Việt dùng DistilBERT ")

# Load model cache: đảm bảo mô hình được tải trước khi sử dụng
load_model()

text = st.text_area("Nhập câu để phân tích", height=150)

if st.button("Phân tích"):
    # Đếm số ký tự (loại bỏ khoảng trắng đầu/cuối)
    char_count = len(text.strip()) 
    MIN_CHARS = 5 # Ngưỡng tối thiểu là 5 ký tự
    
    if not text.strip():
        # Lỗi: Không có văn bản
        st.warning("Vui lòng nhập văn bản để phân tích.")
    elif char_count < MIN_CHARS:
        # Lỗi: Văn bản quá ngắn (dưới 5 ký tự)
        st.error(f"Văn bản quá ngắn ({char_count} ký tự). Vui lòng nhập **ít nhất {MIN_CHARS} ký tự** để phân tích chính xác hơn.")
    else:
        with st.spinner("Đang phân tích và lưu trữ..."):
            # Gọi process_and_store: Hàm này thực hiện TOÀN BỘ quy trình:
            result = process_and_store(text)

            if result and result.get('sentiment'):
                sentiment = result['sentiment']
                
                # ----------------------------------------------------------
                # PHẦN CHỈNH SỬA MÀU SẮC KẾT QUẢ
                # ----------------------------------------------------------
                
                color = ""
                emoji = ""
                
                if sentiment == 'POSITIVE':
                    color = "#00A36C" # Xanh lá (Sử dụng st.success cho màu này)
                   
                elif sentiment == 'NEGATIVE':
                    color = "#FF4B4B" # Đỏ (Sử dụng st.error cho màu này)
                    
                else: # NEUTRAL
                    color = "#F0B90B" # Vàng (Sử dụng st.warning hoặc markdown)
                    

                # Hiển thị kết quả bằng Markdown với màu nền tùy chỉnh (dùng style)
                # Dùng HTML/CSS để định dạng
                html_code = f"""
                <div style="
                    background-color: {color}; 
                    padding: 10px; 
                    border-radius: 5px; 
                    color: white; 
                    font-size: 18px; 
                    font-weight: bold;
                    text-align: center;
                ">
                    {emoji} Cảm xúc: {sentiment}
                </div>
                """
                st.markdown(html_code, unsafe_allow_html=True)
                
                # Hiển thị văn bản đã xử lý
                st.write(f"Văn bản đã được tiền xử lý và lưu vào DB: *{result['text']}*") 
            else:
                # Xử lý trường hợp phân tích thất bại
                st.error("Phân tích thất bại. Vui lòng kiểm tra log lỗi hoặc đảm bảo model/checkpoint đã tải thành công.")

# ----------- LỊCH SỬ -----------
st.subheader("📜 Lịch sử phân loại gần đây")

limit = 10 
offset = st.session_state.get("offset", 0)

# Tải lịch sử
history = load_history(limit=limit, offset=offset)

if history:
    # Hiển thị lịch sử
    for (id, text, sentiment, timestamp) in history:
        # Tùy chỉnh màu sắc cho lịch sử nếu cần thiết (optional)
        if sentiment == 'POSITIVE':
            hist_color = 'green'
        elif sentiment == 'NEGATIVE':
            hist_color = 'red'
        else:
            hist_color = 'orange'
            
        st.markdown(f"**[{timestamp}]** → *{text}* → <span style='color:{hist_color}; font-weight:bold;'>{sentiment}</span>", unsafe_allow_html=True)
else:
    st.info("Chưa có dữ liệu lịch sử nào.")

# Nút tải thêm
if len(history) == limit:
    if st.button("Tải thêm"):
        # Cập nhật offset và chạy lại ứng dụng để tải dữ liệu mới
        st.session_state.offset = offset + limit
        st.rerun()