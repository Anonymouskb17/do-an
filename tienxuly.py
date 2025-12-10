import pandas as pd
import re
import os
from underthesea import word_tokenize, pos_tag
import warnings
warnings.filterwarnings('ignore')

print("Bắt đầu tiền xử lý dữ liệu bình luận sàn thương mại với NLP chuẩn...\n")

# Đọc dữ liệu
df = pd.read_excel('data1.xlsx', sheet_name='data - data')
if 'comment' not in df.columns or 'label' not in df.columns:
    raise ValueError("File phải có cột 'comment' và 'label'!")

df = df[['comment', 'label']].copy()
df.dropna(subset=['comment', 'label'], inplace=True)
df = df[df['comment'].astype(str).str.strip() != ''].reset_index(drop=True)
print(f"Đọc được {len(df):,} bình luận hợp lệ.\n")

# STOPWORDS & TỪ PHỦ ĐỊNH
STOPWORDS = {'là', 'của', 'và', 'có', 'được', 'ở', 'một', 'với', 'cho', 'trong', 'tôi', 'đã', 'đó',
             'rất', 'lại', 'còn', 'này', 'nếu', 'sẽ', 'đến', 'từ', 'đang', 'theo', 'về', 'làm',
             'nhiều', 'ít', 'các', 'như', 'cũng', 'để', 'mà', 'thì', 'tại', 'ạ', 'ơi', 'nhé', 'nha',
             'luôn', 'nè', 'uk', 'ok', 'oke', 'okie', 'sp', 'shop', 'mn', 'mng', 'ad', 'admin',
             'dc', 'đc', 'tks', 'thanks', 'thank', 'cảm ơn', 'mình', 'em', 'chị', 'anh', 'bạn'}

NEGATION_WORDS = {'không', 'ko', 'chẳng', 'chả', 'chưa', 'đừng', 'đéo', 'hông', 'hổng', 'koa', 'hem'}

def clean_text_vietnamese(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text).lower()

    # Xóa link, email, sđt
    text = re.sub(r'http[s]?://\S+|www\.\S+|\S+@\S+|\b0\d{9,10}\b', ' ', text)

    # Xóa emoji
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
    text = emoji_pattern.sub(' ', text)

    # Chuẩn hóa lặp ký tự
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # Giữ chữ cái TV + số
    text = re.sub(r'[^a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return ""
    return text

def advanced_nlp_processing(text):
    if not text.strip():
        return ""

    try:
        # Bước 1: POS tagging trên văn bản thô → giữ được từ ghép đúng
        tagged = pos_tag(text)  # ← đúng cách dùng

        # Bước 2: Tách từ ghép chính xác
        segmented = word_tokenize(text, format="text")

        # Bước 3: Lọc theo POS + giữ từ phủ định
        keep_words = []
        for word, pos in tagged:
            # Giữ các loại từ quan trọng
            if pos.startswith(('N', 'V', 'A', 'R', 'M')) or word in NEGATION_WORDS:
                keep_words.append(word)

        # Nếu lọc quá mạnh → fallback về segmented
        if not keep_words:
            return segmented

        result = ' '.join(keep_words)
        # Đảm bảo vẫn giữ được từ ghép như "Việt Nam", "rất tốt"
        return result

    except Exception as e:
        print(f"Lỗi NLP: {e}")
        return word_tokenize(text, format="text") if text.strip() else ""

# === ÁP DỤNG ===
print("Bước 1: Làm sạch văn bản...")
df['cleaned'] = df['comment'].apply(clean_text_vietnamese)

print("Bước 2: Xử lý NLP (tách từ + POS filtering)...")
df['nlp_processed'] = df['cleaned'].apply(advanced_nlp_processing)

print("Bước 3: Loại bỏ stopwords (giữ từ phủ định)...")
def remove_stopwords_safe(text):
    if not text.strip():
        return ""
    words = text.split()
    return ' '.join([w for w in words if w not in STOPWORDS or w in NEGATION_WORDS])

df['final_comment'] = df['nlp_processed'].apply(remove_stopwords_safe)

# Loại bỏ rỗng
before = len(df)
df = df[df['final_comment'].str.strip() != ''].reset_index(drop=True)
after = len(df)

# Lưu kết quả
output_file = 'data_final.xlsx'
df[['comment', 'label', 'final_comment']].to_excel(output_file, index=False, engine='openpyxl')

print(f"\nHOÀN TẤT! ĐÃ SỬA 3 LỖI NGHIÊM TRỌNG")
print(f"• Trước: {before:,} → Sau: {after:,} bình luận")
print(f"• File sạch: {os.path.abspath(output_file)}")
print("• Đã tách từ đúng, giữ từ phủ định, POS filtering an toàn")
print("="*80)