# sentiment1.py

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
import sys
import os
from typing import Optional

# ---------------------------------------------------------
# 1. IMPORT CÁC MODULE PHỤ TRỢ (Để App hoạt động)
# ---------------------------------------------------------
try:
    from preprocess import preprocess
    from database import save_to_db, load_history
except ImportError as e:
    print(f"Lỗi import: {e}. Đang chạy chế độ debug/độc lập.")
    def preprocess(text): return text
    def save_to_db(text, sentiment): print(f"DB Simulated: {text} -> {sentiment}")
    def load_history(limit, offset): return []

# ---------------------------------------------------------
# 2. CẤU HÌNH (CONFIG)
# ---------------------------------------------------------
checkpoint_path = "checkpoint.pth" 
model_name = "distilbert-base-multilingual-cased"
device = torch.device("cpu")
CONFIDENCE_THRESHOLD = 0.5

ID_TO_LABEL = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
LABEL_TO_ID = {"NEGATIVE": 0, "POSITIVE": 1, "NEUTRAL": 2}

_MODEL_CACHED = None

def _load_model_cached():
    """
    Hàm tải model và tokenizer một lần duy nhất.
    Kích hoạt fix lỗi 'module.' và resize token embeddings.
    """
    global _MODEL_CACHED
    if _MODEL_CACHED is not None:
        return _MODEL_CACHED

    print(f"--- Bắt đầu tải Model: {model_name} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        new_tokens = ["<e>", "</e>"]
        tokenizer.add_tokens(new_tokens)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(ID_TO_LABEL)
        )


        model.resize_token_embeddings(len(tokenizer))

        if os.path.exists(checkpoint_path):
            print(f"Đang tải checkpoint từ: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint


            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(new_state_dict)
            print("Đã load weights thành công!")
        else:
            print(f"CẢNH BÁO: Không tìm thấy file {checkpoint_path}. Dùng weight random.")

        model.to(device)
        model.eval()

        _MODEL_CACHED = (model, tokenizer)
        return model, tokenizer

    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG KHI LOAD MODEL: {e}")
        return None, None


def classify_text(text: str) -> dict:
    """
    Hàm phân loại cảm xúc.
    """
    model, tokenizer = _load_model_cached()

    if model is None or tokenizer is None:
        return {"sentiment": "UNKNOWN", "label_id": -1, "max_prob": 0.0}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probs = torch.softmax(logits, dim=1)
    max_prob, predicted_id = torch.max(probs, 1)

    label_id = predicted_id.item()
    max_prob_value = max_prob.item()
    
    sentiment = ID_TO_LABEL.get(label_id, "UNKNOWN")
    
    return {
        "sentiment": sentiment,
        "label_id": label_id,
        "max_prob": max_prob_value,
        "logits": logits.tolist()
    }


def process_and_store(raw_text: str) -> Optional[dict]:
    """
    Pipeline chính được gọi bởi Streamlit (app.py):
    1. Preprocess
    2. Classify
    3. Save DB
    """
    if not raw_text or not raw_text.strip():
        return None

    processed_text = preprocess(raw_text)

    result = classify_text(processed_text)

    if result["sentiment"] == "UNKNOWN":
        return None
    
    save_to_db(raw_text, result["sentiment"])

    return {
        "raw_text": raw_text,
        "text": processed_text, # Text đã xử lý để hiển thị debug nếu cần
        "sentiment": result["sentiment"],
        "probability": result["max_prob"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

