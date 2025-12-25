import os
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Kết quả dự đoán", layout="wide")
st.title("Kết quả dự đoán")  # ✅ không nhắc tập test

MODEL_PATH = "ecommerce_sentiment_model.joblib"
DATA_PATH = "data_final.xlsx"
SHEET_NAME = "Sheet1"

@st.cache_resource
def load_model():
    return load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    for col in ["final_comment", "label"]:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột '{col}' trong {DATA_PATH}")

    if "comment" not in df.columns:
        df["comment"] = df["final_comment"]

    df["final_comment"] = df["final_comment"].fillna("").astype(str)
    return df

if not os.path.exists(MODEL_PATH):
    st.error(f"Không thấy file model: {MODEL_PATH}")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error(f"Không thấy file dữ liệu: {DATA_PATH}")
    st.stop()

model = load_model()
df = load_data()

X = df["final_comment"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if st.button("Chạy và hiển thị kết quả", use_container_width=True):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.subheader("Kết quả đánh giá")
    st.write(f"Train size: **{len(X_train):,}** | Test size: **{len(X_test):,}**")
    st.write(f"Accuracy: **{acc:.4f}**")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Bảng dự đoán")  # ✅ không nhắc test
    test_results = pd.DataFrame({
        "comment": df.loc[X_test.index, "comment"].astype(str).values,
        "final_comment": X_test.values,
        "true": y_test.values,
        "pred": y_pred
    })
    st.dataframe(test_results, use_container_width=True)
