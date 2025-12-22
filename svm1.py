
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

    # ĐỌC ĐÚNG SHEET 'data' VÀ DÙNG CỘT final_comment
    try:
        df = pd.read_excel('data_final.xlsx', sheet_name='Sheet1')
    except FileNotFoundError:
        raise FileNotFoundError("Không tìm thấy file 'data_final.xlsx'. Hãy chạy tiền xử lý trước (tienxuly.py) hoặc đặt file vào thư mục hiện tại.")

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
        SVC(kernel='linear', C=1.2, random_state=42, class_weight='balanced')
    )

    print("Đang huấn luyện mô hình SVM + TF-IDF (có thể mất 1-3 phút)...")
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_test)

    # Safe probability extraction
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X_test).max(axis=1)
        except Exception:
            probs = np.full(len(y_pred), np.nan)
    else:
        probs = np.full(len(y_pred), np.nan)

    if args.show_test:
        acc = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 70)
        print("KẾT QUẢ HUẤN LUYỆN - PHÂN TÍCH CẢM XÚC ")
        print("=" * 70)
        print(f"Accuracy: {acc:.4f} ({acc:.2%})")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("=" * 70)

        # Hiển thị 10 ví dụ thực tế
        print("\n10 DỰ ĐOÁN THỰC TẾ (ngẫu nhiên từ tập test):")
        print("-" * 70)

        test_results = pd.DataFrame({
            'comment': df.loc[X_test.index, 'comment'],
            'true': y_test,
            'pred': y_pred,
            'prob': probs
        }).reset_index(drop=True)

        np.random.seed(42)
        samples = test_results.sample(10)

        for i, row in samples.iterrows():
            status = "ĐÚNG" if row['true'] == row['pred'] else "SAI"
            prob_str = f" (độ tin cậy: {row['prob']:.3f})" if not np.isnan(row['prob']) else ""
            print(f"[{status}] -> {row['pred']}" + prob_str)
            text = str(row['comment'])
            print(f"   {text[:120]}{'...' if len(text) > 120 else ''}\n")

    else:
        # still construct test_results for optional saving
        test_results = pd.DataFrame({
            'comment': df.loc[X_test.index, 'comment'],
            'true': y_test,
            'pred': y_pred,
            'prob': probs
        }).reset_index(drop=True)

    # Optionally save test results
    if args.save_test_csv:
        try:
            test_results.to_csv(args.save_test_csv, index=False, encoding='utf-8-sig')
            print(f"Test predictions saved to: {os.path.abspath(args.save_test_csv)}")
        except Exception as e:
            print(f"Không thể lưu test results: {e}")

    # Lưu mô hình
    model_file = args.model_out
    dump(model, model_file)
    print(f"\nMÔ HÌNH ĐÃ ĐƯỢC LƯU: {os.path.abspath(model_file)}")


if __name__ == '__main__':
    main()
