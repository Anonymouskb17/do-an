
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os

print("Đang đọc dữ liệu đã làm sạch...\n")

# ĐỌC ĐÚNG SHEET 'data' VÀ DÙNG CỘT final_comment
df = pd.read_excel('data_final.xlsx', sheet_name='data')

print(f"Đã tải {len(df):,} bình luận đã được xử lý sạch.")
print("Phân bố nhãn:")
print(df['label'].value_counts())
print()

# DÙNG CỘT ĐÃ LÀM SẠCH HOÀN TOÀN
X = df['final_comment'].fillna("")
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline tối ưu cho văn bản đã sạch
model = make_pipeline(
    TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        lowercase=False  
    ),
    SVC(kernel='linear', E=1.2, random_state=42, class_weight='balanced')
)

print("Đang huấn luyện mô hình SVM + TF-IDF (có thể mất 1-3 phút)...")
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("KẾT QUẢ HUẤN LUYỆN - PHÂN TÍCH CẢM XÚC ")
print("="*70)
print(f"Accuracy: {acc:.4f} ({acc:.2%})")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("="*70)

# Hiển thị 10 ví dụ thực tế
print("\n10 DỰ ĐOÁN THỰC TẾ (ngẫu nhiên từ tập test):")
print("-"*70)

test_results = pd.DataFrame({
    'comment': df.loc[X_test.index, 'comment'],
    'true': y_test,
    'pred': y_pred,
    'prob': model.predict_proba(X_test).max(axis=1)
}).reset_index(drop=True)

np.random.seed(42)
samples = test_results.sample(10)

for i, row in samples.iterrows():
    status = "ĐÚNG" if row['true'] == row['pred'] else "SAI"
    print(f"[{status}] → {row['pred']} ")
    text = str(row['comment'])
    print(f"   {text[:120]}{'...' if len(text)>120 else ''}\n")

# Lưu mô hình
model_file = 'shopee_sentiment_model.joblib'
dump(model, model_file)
print(f"\nMÔ HÌNH ĐÃ ĐƯỢC LƯU: {os.path.abspath(model_file)}")
