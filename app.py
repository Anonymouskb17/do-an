import streamlit as st
from joblib import load


st.set_page_config(page_title="Phân tích cảm xúc Shopee", layout="centered")
st.title("Phân tích cảm xúc bình luận Shopee")
st.markdown("Nhập bình luận → nhấn nút → xem kết quả ngay")


@st.cache_resource
def get_model():
    return load("shopee_sentiment_model.joblib")

model = get_model()


user_input = st.text_area(
    "Nhập bình luận của bạn ở đây",
    height=150,
    placeholder="ví dụ: hàng đẹp lắm shop ơi, giao nhanh nữa..."
)

if st.button("Dự đoán cảm xúc", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Bạn chưa nhập bình luận nào!")
    else:
        with st.spinner("Đang phân tích..."):
            pred = model.predict([user_input])[0]
            prob = model.predict_proba([user_input]).max()

        # Hiển thị kết quả đẹp
        if pred == "POS":
            st.success(f"TÍCH CỰC  (độ tin cậy: {prob:.1%})")
        elif pred == "NEG":
            st.error(f"TIÊU CỰC  (độ tin cậy: {prob:.1%})")
        else:  # NEU
            st.info(f"TRUNG LẬP  (độ tin cậy: {prob:.1%})")

        # Thanh độ tin cậy
        st.progress(prob)

st.markdown("---")
st.caption("Mô hình SVM + TF-IDF – huấn luyện trên >30k bình luận Shopee thực tế")