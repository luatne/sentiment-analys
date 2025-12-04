import sqlite3
from datetime import datetime

DB_NAME = "sentiment.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Tạo bảng (phiên bản mới có timestamp)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    # Sau khi đảm bảo bảng tồn tại, chạy migration để thêm cột nếu bảng cũ không có
    migrate_add_timestamp_if_missing()


def migrate_add_timestamp_if_missing():
    """
    Kiểm tra nếu cột 'timestamp' chưa có thì thêm cột và cập nhật các bản ghi cũ
    bằng timestamp hiện tại (local time).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Lấy thông tin cột
    cursor.execute("PRAGMA table_info(sentiments);")
    cols = [row[1] for row in cursor.fetchall()]  # row[1] là tên cột

    if "timestamp" not in cols:
        # Thêm cột timestamp
        cursor.execute("ALTER TABLE sentiments ADD COLUMN timestamp TEXT;")
        # Cập nhật các bản ghi cũ (nếu có) với timestamp hiện tại localtime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("UPDATE sentiments SET timestamp = ? WHERE timestamp IS NULL OR timestamp = ''", (now,))
        conn.commit()

    conn.close()


def save_to_db(text, sentiment):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Parameterized query → chống SQL injection
    cursor.execute("""
        INSERT INTO sentiments (text, sentiment, timestamp)
        VALUES (?, ?, ?)
    """, (text, sentiment, timestamp))

    conn.commit()
    conn.close()


def load_history(limit=50, offset=0):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, text, sentiment, timestamp 
        FROM sentiments
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    rows = cursor.fetchall()
    conn.close()
    return rows
