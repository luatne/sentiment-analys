from underthesea import word_tokenize
import re
import os # Cần import os để kiểm tra sự tồn tại của file

# Tên file chứa teencode (Giống với tên file đã tạo: teencode.txt)
TEENCODE_FILE = "teencode.txt"

# Dictionary sửa lỗi, viết tắt phổ biến (Sẽ được tải từ file)
teencode = {}

# Hàm tải dictionary từ file
def load_teencode_dict(filepath):
    """
    Tải dictionary teencode từ file văn bản.
    File phải có định dạng: <từ viết tắt>\t<từ đầy đủ> trên mỗi dòng (tách bằng tab).
    """
    # In ra thông báo để biết hàm đang hoạt động
    print(f"Đang tải teencode từ file: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Lỗi: File {filepath} không tồn tại. Sử dụng dictionary rỗng.")
        return {}
    
    loaded_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Loại bỏ khoảng trắng đầu/cuối và tách bằng tab
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    # Lấy từ viết tắt và từ đầy đủ, chuyển thành chữ thường
                    abbr = parts[0].strip().lower()
                    full = parts[1].strip().lower()
                    if abbr and full:
                        loaded_dict[abbr] = full
        print(f"Đã tải thành công {len(loaded_dict)} cặp teencode từ {filepath}.")
    except Exception as e:
        print(f"Lỗi khi đọc file {filepath}: {e}")
        return {}
        
    return loaded_dict

# Tải dictionary ngay khi module được import
teencode = load_teencode_dict(TEENCODE_FILE)


def preprocess(text):
    """
    Thực hiện tiền xử lý văn bản: chuyển chữ thường, thay thế teencode,
    loại bỏ ký tự đặc biệt, tách từ (word_tokenize) và giới hạn độ dài.
    """
    # 1. Chuyển chữ thường
    text = text.lower()

    # 2. Thay từ viết tắt (Sử dụng dictionary đã tải từ file teencode.txt)
    # Đảm bảo thay thế toàn bộ từ (dùng \b để khớp với ranh giới từ)
    if teencode: # Chỉ chạy nếu dictionary đã được tải
        for abbr, full in teencode.items():
            # Thoát ký tự đặc biệt trong từ viết tắt để dùng trong Regex
            escaped_abbr = re.escape(abbr)
            
            # Sử dụng \b để đảm bảo chỉ thay thế toàn bộ từ viết tắt (word boundary)
            # Dùng try/except cho re.sub để tránh lỗi nếu có vấn đề với regex pattern phức tạp
            try:
                text = re.sub(rf"\b{escaped_abbr}\b", full, text)
            except re.error as e:
                # Fallback: nếu lỗi regex, dùng thay thế chuỗi đơn giản (ít chính xác hơn)
                # Tuy nhiên, do re.escape() đã được dùng, lỗi này ít xảy ra.
                print(f"Regex Error cho {abbr}: {e}. Dùng str.replace.")
                text = text.replace(abbr, full)


    # 3. Loại ký tự đặc biệt không cần thiết
    # Giữ lại chữ cái tiếng Việt, chữ cái Latin, số, và khoảng trắng
    text = re.sub(r"[^a-zA-ZÀ-Ỳà-ỳ0-9\s]", " ", text)

    # 4. Tách từ bằng underthesea (Vietnamese Word Tokenization)
    text = word_tokenize(text, format="text")

    # 5. Giới hạn câu ≤ 50 từ (Nếu cần để đảm bảo input phù hợp với model)
    words = text.split()
    if len(words) > 50:
        words = words[:50]

    return " ".join(words)