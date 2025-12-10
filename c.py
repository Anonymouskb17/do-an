from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Fake data (thay bằng dữ liệu của bạn)
texts = [
    "Tôi rất thích sản phẩm này",
    "Quá tệ, thất vọng",
    "Bình thường, không có gì đặc biệt",
    "Xuất sắc, quá tuyệt vời",
    "Sản phẩm quá xấu",
    "Không biết nữa, thấy cũng ổn"
]
labels = ["positive", "negative", "neutral", "positive", "negative", "neutral"]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

C_values = [0.1, 0.3, 1, 5, 10, 100]

results = {}

for C in C_values:
    model = make_pipeline(
        TfidfVectorizer(max_features=2000, ngram_range=(1,2)),
        SVC(kernel='linear', C=C)
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    results[C] = acc
    
    print(f"\n===== C = {C} =====")
    print(classification_report(y_test, preds))

print("\nSummary accuracy:")
for C, acc in results.items():
    print(f"C={C}: accuracy={acc:.4f}")
