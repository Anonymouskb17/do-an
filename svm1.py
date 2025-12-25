import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os


def main():
    parser = argparse.ArgumentParser(description='Train SVM TF-IDF sentiment model')
    parser.add_argument('--show-test', action='store_true', help='Show test-set metrics and sample predictions')
    parser.add_argument('--save-test-csv', type=str, default=None, help='Path to save test predictions CSV')
    parser.add_argument('--model-out', type=str, default='ecommerce_sentiment_model.joblib', help='Path to save trained model')
    args = parser.parse_args()

    print("Đang đọc dữ liệu đã làm sạch...\n")

    try:
        df = pd.read_excel('data_final.xlsx', sheet_name='Sheet1')  # đổi nếu sheet khác
    except FileNotFoundError:
        raise FileNotFoundError(
            "Không tìm thấy file 'data_final.xlsx'. Hãy chạy tiền xử lý trước (tienxuly.py) "
            "hoặc đặt file vào thư mục hiện tại."
        )

    print(f"Đã tải {len(df):,} bình luận đã được xử lý sạch.")
    print("Phân bố nhãn:")
    print(df['label'].value_counts())
    print()

    # X: văn bản đã làm sạch, y: nhãn
    X = df['final_comment'].fillna("")
    y = df['label']

  
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,        
        random_state=42,
        stratify=y            
    )

    
    
    model = make_pipeline(
        TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 4),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            lowercase=False
        ),
        SVC(kernel='poly', C=1.2, random_state=42, class_weight='balanced')
    )

    print("Đang huấn luyện mô hình SVM + TF-IDF (train 80%) ...")
    model.fit(X_train, y_train)   

    print("Đang đánh giá mô hình trên tập test 20% ...")
    y_pred = model.predict(X_test)  

    # Tính accuracy và report
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ (TEST 20%) - SVM + TF-IDF")
    print("=" * 70)
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Accuracy: {acc:.4f} ({acc:.2%})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("=" * 70)

    # Tạo bảng kết quả test để lưu / xem mẫu
    test_results = pd.DataFrame({
        'comment': df.loc[X_test.index, 'comment'],
        'final_comment': X_test.values,
        'true': y_test.values,
        'pred': y_pred
    }).reset_index(drop=True)

    # In 10 mẫu dự đoán nếu cần
    if args.show_test:
        print("\n10 DỰ ĐOÁN THỰC TẾ (ngẫu nhiên từ tập test 20%):")
        print("-" * 70)

        np.random.seed(42)
        samples = test_results.sample(min(10, len(test_results)))

        for _, row in samples.iterrows():
            status = "ĐÚNG" if row['true'] == row['pred'] else "SAI"
            print(f"[{status}] true={row['true']} -> pred={row['pred']}")
            text = str(row['comment'])
            print(f"   {text[:120]}{'...' if len(text) > 120 else ''}\n")

    # Lưu file test predictions nếu user yêu cầu
    if args.save_test_csv:
        try:
            test_results.to_csv(args.save_test_csv, index=False, encoding='utf-8-sig')
            print(f"Test predictions saved to: {os.path.abspath(args.save_test_csv)}")
        except Exception as e:
            print(f"Không thể lưu test results: {e}")

    # Lưu mô hình
    dump(model, args.model_out)
    print(f"\nMÔ HÌNH ĐÃ ĐƯỢC LƯU: {os.path.abspath(args.model_out)}")


if __name__ == '__main__':
    main()
