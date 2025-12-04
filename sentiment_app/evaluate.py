import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
import sys
import os 

# ----------------------------------------------------------------------
# IMPORT TỪ CÁC MODULE KHÁC
# Sửa: Loại bỏ LABEL_MAP vì không tồn tại trong sentiment1.py
# ----------------------------------------------------------------------
try:
    from sentiment1 import _load_model_cached, classify_text 
    from preprocess import preprocess
except ImportError as e:
    print(f"Lỗi: Không thể import sentiment1.py, preprocess.py. Chi tiết: {e}")
    print("Vui lòng đảm bảo các file sentiment1.py và preprocess.py đã được tạo và nằm trong cùng thư mục.")
    sys.exit(1)

# ----------------------------------------------------------------------
# BỘ DATASET KIỂM THỬ THỦ CÔNG (10 CÂU)
# ----------------------------------------------------------------------

# Dữ liệu: [Văn bản, Nhãn đúng (Positive, Negative, Neutral)]
TEST_DATA = [
    # Positive (Tích cực)
    ("Hôm nay tôi rất vui  ", "POSITIVE"), # Chuyển thành chữ hoa để khớp với sentiment1.py
    ("rat vui hom nay.", "POSITIVE"),
    ("phim này hay lắm", "POSITIVE"),
    ("cảm ơn bạn rất nhiều", "POSITIVE"),
    
    # Negative (Tiêu cực)
    ("món ăn này dở quá", "NEGATIVE"),
    ("tôi buồn vì thất bại", "NEGATIVE"),
    ("mệt mỏi quá hôm nay", "NEGATIVE"),

    # Neutral (Trung lập)
    ("thời tiết bình thường", "NEUTRAL"),
    ("công việc ổn định", "NEUTRAL"),
    ("ngày mai đi học", "NEUTRAL"),
]

def classify_only(raw_text: str):
    """
    Tiền xử lý và phân loại văn bản, chỉ trả về kết quả sentiment.
    """
    if not raw_text or not raw_text.strip():
        return "N/A"
    
    # Tiền xử lý văn bản
    processed_text = preprocess(raw_text)
    
    # Phân loại
    result = classify_text(processed_text)
    
    if result is None:
        return "ERROR"
        
    return result['sentiment']

# ----------------------------------------------------------------------
# HÀM CHÍNH: CHẠY ĐÁNH GIÁ TRÊN 10 CÂU TEST
# ----------------------------------------------------------------------
def evaluate_manual_test_set(test_data):
    """
    Thực hiện phân loại trên bộ 10 câu test thủ công và tính toán metrics.
    """
    print(">>> TẢI MODEL ĐỂ ĐÁNH GIÁ (Sẽ chạy nhanh nếu model đã được cache) <<<")
    # Kích hoạt tải model (chạy 1 lần)
    model, tokenizer = _load_model_cached()
    
    if model is None or tokenizer is None:
        print("Lỗi: Không thể tải model. Dừng đánh giá.")
        return

    all_predictions = []
    all_true_labels = []
    detailed_results = []
    
    print("\n>>> BẮT ĐẦU ĐÁNH GIÁ TRÊN BỘ 10 CÂU TEST <<<")
    
    # Cần đảm bảo nhãn đúng trong TEST_DATA khớp với chuỗi trả về từ classify_only
    # Vì sentiment1.py trả về 'POSITIVE', 'NEGATIVE', 'NEUTRAL' (chữ hoa),
    # tôi đã sửa TEST_DATA ở trên để khớp.

    for text, true_label in tqdm(test_data, desc="Đang dự đoán"):
        predicted_sentiment = classify_only(text)
        
        # Chỉ thu thập các mẫu có kết quả hợp lệ
        if predicted_sentiment not in ["N/A", "ERROR"]:
            all_predictions.append(predicted_sentiment)
            all_true_labels.append(true_label)
            # So sánh chuỗi nhãn
            is_correct = "✅ Đúng" if predicted_sentiment == true_label else "❌ Sai"
            
            detailed_results.append({
                "Văn bản": text,
                "Nhãn đúng": true_label,
                "Dự đoán": predicted_sentiment,
                "Kết quả": is_correct
            })
    
    # In kết quả chi tiết
    print("\n--- KẾT QUẢ DỰ ĐOÁN CHI TIẾT ---")
    df = pd.DataFrame(detailed_results)
    print(df.to_markdown(index=False)) 

    # ----------------------------------------------------
    # TÍNH ACCURACY VÀ METRICS
    # ----------------------------------------------------
    
    print("\n===================================================\n")
    if not all_predictions:
        print("Không có mẫu hợp lệ nào được thu thập. Vui lòng kiểm tra lại cấu hình.")
        return

    # Tính toán Accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    # Lấy danh sách nhãn duy nhất có trong dữ liệu test và sắp xếp
    unique_labels = sorted(list(set(all_true_labels))) 

    print(f">>> KẾT QUẢ ĐÁNH GIÁ TRÊN {len(TEST_DATA)} CÂU TEST THỦ CÔNG <<<")
    print(f"✅ ĐỘ CHÍNH XÁC (ACCURACY): {accuracy:.4f}")
    
    print("\n--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
    cm = confusion_matrix(all_true_labels, all_predictions, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=[f'True: {l}' for l in unique_labels], columns=[f'Pred: {l}' for l in unique_labels])
    print(cm_df.to_markdown())

    print("\n--- BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ---")
    # Lưu ý: Cần đảm bảo các nhãn trong unique_labels khớp với nhãn trong báo cáo
    report = classification_report(all_true_labels, all_predictions, labels=unique_labels)
    print(report)
    print("===================================================\n")


if __name__ == '__main__':
    evaluate_manual_test_set(TEST_DATA)