import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os


def main():
    parser = argparse.ArgumentParser(description='Train Logistic TF-IDF sentiment model')
    parser.add_argument('--save-test-csv', type=str, default=None, help='Path to save test predictions CSV')
    parser.add_argument('--model-out', type=str, default='ecommerce_logistic_sentiment_model.joblib',
                        help='Path to save trained model')
    args = parser.parse_args()

    print("Đang đọc dữ liệu đã làm sạch...\n")

    try:
        df = pd.read_excel('data_final.xlsx', sheet_name='Sheet1')
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
    X = df['final_comment'].fillna("").astype(str)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
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
            lowercase=False
        ),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            multi_class="auto"
        )
    )

    print("Đang huấn luyện mô hình ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Accuracy: {acc:.4f} ({acc:.2%})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("=" * 70)

    # Bảng kết quả test để lưu (không in ra)
    test_results = pd.DataFrame({
        'comment': df.loc[X_test.index, 'comment'] if 'comment' in df.columns else X_test.values,
        'final_comment': X_test.values,
        'true': y_test.values,
        'pred': y_pred
    }).reset_index(drop=True)

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
