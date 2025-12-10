import os
import numpy as np
import streamlit as st
from joblib import load


st.set_page_config(page_title="Phân tích cảm xúc sàn thương mại", layout="centered")
st.title("Phân tích cảm xúc bình luận sàn thương mại")
st.markdown("Nhập bình luận → nhấn nút → xem kết quả ngay")


@st.cache_resource
def get_model():
    # Thử nhiều tên file model (fallback)
    candidates = [
        'ecommerce_sentiment_model.joblib',
        'shopee_sentiment_model.joblib'
    ]
    for p in candidates:
        if os.path.exists(p):
            return load(p)
    raise FileNotFoundError(
        f"Không tìm thấy file mô hình. Kiểm tra các tệp: {candidates} hoặc huấn luyện mô hình bằng `svm1.py`.")


try:
    model = get_model()
except Exception as e:
    st.error(str(e))
    model = None


user_input = st.text_area(
    "Nhập bình luận của bạn ở đây",
    height=150,
    placeholder="ví dụ: hàng đẹp lắm, giao nhanh nữa..."
)

if st.button("Dự đoán cảm xúc", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Bạn chưa nhập bình luận nào!")
    elif model is None:
        st.error("Mô hình chưa nạp. Kiểm tra log lỗi phía trên hoặc chạy `svm1.py` để huấn luyện và lưu mô hình.")
    else:
        with st.spinner("Đang phân tích..."):
            pred = model.predict([user_input])[0]
            # Lấy xác suất một cách an toàn
            if hasattr(model, 'predict_proba'):
                try:
                    prob = model.predict_proba([user_input]).max()
                except Exception:
                    prob = float('nan')
            else:
                prob = float('nan')

        # Hiển thị kết quả đẹp
        prob_display = f" (độ tin cậy: {prob:.1%})" if not np.isnan(prob) else ""
        if pred == "POS":
            st.success(f"TÍCH CỰC{prob_display}")
        elif pred == "NEG":
            st.error(f"TIÊU CỰC{prob_display}")
        else:  # NEU
            st.info(f"TRUNG LẬP{prob_display}")

        # Thanh độ tin cậy (nếu có)
        if not np.isnan(prob):
            try:
                st.progress(float(prob))
            except Exception:
                pass

st.markdown("---")
st.caption("Mô hình SVM + TF-IDF – huấn luyện trên >30k bình luận sàn thương mại thực tế")