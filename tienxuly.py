import pandas as pd
import re
import os
from underthesea import word_tokenize
import warnings
warnings.filterwarnings('ignore')

print("Bắt đầu tiền xử lý (NLP tiếng Việt - tối ưu cho train SVM TF-IDF)...\n")

# =========================
# 1) Đọc dữ liệu
# =========================
df = pd.read_excel('data1.xlsx', sheet_name='data - data')
if 'comment' not in df.columns or 'label' not in df.columns:
    raise ValueError("File phải có cột 'comment' và 'label'!")

df = df[['comment', 'label']].copy()
df.dropna(subset=['comment', 'label'], inplace=True)
df = df[df['comment'].astype(str).str.strip() != ''].reset_index(drop=True)
print(f"Đọc được {len(df):,} bình luận hợp lệ.\n")

# =========================
# 2) Từ phủ định & từ đảo chiều (bắt buộc giữ)
# =========================
NEGATION_WORDS = {
    "không", "chẳng", "chả", "chưa", "đừng", "hông", "hổng", "hem", "đéo"
}
CONTRAST_WORDS = {
    "nhưng", "tuy", "tuy_nhiên", "mặc_dù", "dù", "mà"
}

# =========================
# 3) Stopwords nhẹ (KHÔNG bỏ từ cảm xúc như: tốt, tệ, ok, cảm_ơn, rất, quá...)
# =========================
STOPWORDS = {
    "là", "của", "và", "có", "được", "ở", "một", "với", "cho", "trong", "tôi", "đã",
    "đó", "này", "nếu", "sẽ", "đến", "từ", "đang", "theo", "về", "làm", "các", "như",
    "cũng", "để", "thì", "tại", "ạ", "ơi", "nhé", "nha", "luôn", "nè"
}

# =========================
# 4) Chuẩn hóa teencode cơ bản (để sentiment học tốt hơn)
# =========================
REPLACE_MAP = {
    r"\bko\b": "không",
    r"\bk\b": "không",
    r"\bkh\b": "không",
    r"\bhok\b": "không",
    r"\bhong\b": "không",
    r"\bhông\b": "không",
    r"\bhok\b": "không",
    r"\bhem\b": "không",
    r"\bkg\b": "không",

    r"\btks\b": "cảm_ơn",
    r"\bthanks\b": "cảm_ơn",
    r"\bthank\b": "cảm_ơn",
    r"\btk\b": "cảm_ơn",

    r"\boke\b": "ok",
    r"\bokie\b": "ok",
    r"\bokela\b": "ok",
}

def clean_text(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text).lower()

    # Xóa link, email, sđt
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+|\b0\d{9,10}\b', ' ', text)

    # Xóa emoji (theo yêu cầu)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)

    # Chuẩn hóa lặp ký tự (đẹppppp -> đẹpp)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # Chuẩn hóa viết tắt/teencode trước khi lọc ký tự
    for pattern, repl in REPLACE_MAP.items():
        text = re.sub(pattern, repl, text)

    # Giữ chữ Việt + số + khoảng trắng + dấu gạch dưới (để giữ cảm_ơn, tuy_nhiên)
    text = re.sub(
        r"[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
        r"óòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9\s_]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_stopwords(text: str) -> str:
    if not text.strip():
        return ""

    # Tokenize chuẩn tiếng Việt
    tok = word_tokenize(text, format="text")
    words = tok.split()

    keep = []
    for w in words:
        # Luôn giữ phủ định & từ đảo chiều
        if w in NEGATION_WORDS or w in CONTRAST_WORDS:
            keep.append(w)
            continue

        # Bỏ stopwords nhẹ
        if w in STOPWORDS:
            continue

        # Bỏ token 1 ký tự vô nghĩa (nhưng giữ 'ok' nếu có)
        if len(w) == 1 and w not in {"ok"}:
            continue

        keep.append(w)

    return " ".join(keep).strip()

print("Bước 1: Làm sạch (link/email/sđt + emoji)...")
df["cleaned"] = df["comment"].apply(clean_text)

print("Bước 2: NLP tokenize + stopwords nhẹ (giữ phủ định, đảo chiều)...")
df["final_comment"] = df["cleaned"].apply(tokenize_stopwords)

before = len(df)
df = df[df["final_comment"].str.strip() != ""].reset_index(drop=True)
after = len(df)

output_file = "data_final.xlsx"
df[["comment", "label", "final_comment"]].to_excel(output_file, index=False, engine="openpyxl")

print("\nHOÀN TẤT!")
print(f"• Trước: {before:,} → Sau: {after:,} bình luận")
print(f"• File: {os.path.abspath(output_file)}")
print("=" * 80)
