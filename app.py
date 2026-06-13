import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB
# ==============================================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ==============================================================================
# 2. TẢI MÔ HÌNH VÀ BỘ CHUẨN HÓA (SCALER)
# ==============================================================================
@st.cache_resource
def load_assets():
    try:
        # Tải mô hình đã được huấn luyện
        model = joblib.load("fraud_model.pkl")
        
        # Tải bộ chuẩn hóa dữ liệu để đồng bộ thang đo
        scaler = joblib.load("scaler.pkl")
        
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ Thiếu file hệ thống: Không tìm thấy file `{e.filename}`.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Lỗi không xác định khi tải file: {e}")
        st.stop()

model, scaler = load_assets()

# ==============================================================================
# 3. GIAO DIỆN VÀ THANH ĐIỀU KHIỂN (SIDEBAR)
# ==============================================================================
st.title("💳 Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng")

st.sidebar.header("⚙️ Cấu hình bộ lọc rủi ro")

# Thanh trượt độ nhạy
threshold = st.sidebar.slider(
    "Ngưỡng quyết định gian lận (Threshold)",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
)

# ==============================================================================
# 4. XỬ LÝ TẬP TIN DỮ LIỆU TẢI LÊN
# ==============================================================================
uploaded_file = st.file_uploader(
    "Tải lên tập tin .csv chứa dữ liệu giao dịch cần kiểm tra",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Đọc dữ liệu đầu vào
        df = pd.read_csv(uploaded_file)
        results_df = df.copy()
        
        # Loại bỏ các cột kết quả có sẵn trong file gốc (nếu có)
        feature_cols = [col for col in df.columns if col not in ["Class", "Prediction"]]
        X = df[feature_cols]
        
        # ----------------------------------------------------------------------
        # BƯỚC QUAN TRỌNG: CHUẨN HÓA (SCALE) DỮ LIỆU
        # ----------------------------------------------------------------------
        try:
            # Kiểm tra xem scaler từ Notebook được huấn luyện trên những cột nào
            if hasattr(scaler, "feature_names_in_"):
                scaled_cols = scaler.feature_names_in_
                X_to_scale = X[scaled_cols]
                X[scaled_cols] = scaler.transform(X_to_scale)
            else:
                # Fallback nếu scaler phiên bản cũ: Scale toàn bộ tập X
                X_scaled_array = scaler.transform(X)
                X = pd.DataFrame(X_scaled_array, columns=X.columns)
        except Exception as scale_err:
            st.error(f"❌ Lỗi chuẩn hóa dữ liệu: Cột trong file .csv không khớp với lúc huấn luyện. Chi tiết: {scale_err}")
            st.stop()

        # ----------------------------------------------------------------------
        # CHẠY MÔ HÌNH DỰ ĐOÁN
        # ----------------------------------------------------------------------
        # Dự đoán xác suất (Probability) để kết hợp với thanh trượt Threshold
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            results_df["Fraud_Probability"] = probabilities
        else:
            predictions = model.predict(X)
            results_df["Fraud_Probability"] = predictions.astype(float)
            st.sidebar.warning("⚠️ Mô hình hiện tại không hỗ trợ xác suất. Ngưỡng điều chỉnh tạm vô hiệu.")

        results_df["Prediction"] = predictions

        # Tính toán số liệu thống kê
        total_transactions = len(results_df)
        fraud_count = int(np.sum(predictions == 1))
        normal_count = total_transactions - fraud_count
        fraud_percentage = (fraud_count / total_transactions) * 100

        # ==============================================================================
        # 5. HIỂN THỊ KẾT QUẢ LÊN DASHBOARD
        # ==============================================================================
        st.subheader("📊 Thống Kê Dự Đoán (Prediction Summary)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Tổng số giao dịch", value=f"{total_transactions:,}")
        with col2:
            st.metric(label="Giao dịch bình thường", value=f"{normal_count:,}")
        with col3:
            st.metric(label="Giao dịch gian lận", value=f"{fraud_count:,}")
        with col4:
            st.metric(label="Tỷ lệ gian lận dự đoán", value=f"{fraud_percentage:.4f}%")

        # Danh sách giao dịch bị phát hiện
        st.subheader("🚨 Danh Sách Giao Dịch Nghi Ngờ Gian Lận")
        fraud_df = results_df[results_df["Prediction"] == 1]

        if len(fraud_df) > 0:
            # Sắp xếp các ca gian lận từ mức độ rủi ro cao nhất xuống thấp nhất
            if "Fraud_Probability" in fraud_df.columns:
                fraud_df = fraud_df.sort_values(by="Fraud_Probability", ascending=False)
            
            st.write(f"⚠️ Phát hiện **{len(fraud_df):,}** giao dịch vượt ngưỡng rủi ro `{threshold}`:")
            st.dataframe(fraud_df.head(100)) # Chỉ in tối đa 100 dòng để web không bị đơ
        else:
            st.success("✅ Tuyệt vời! Không phát hiện giao dịch gian lận nào.")

        # Nút xem toàn bộ
        with st.expander("👁️ Xem chi tiết bảng kết quả đầy đủ"):
            st.dataframe(results_df.head(100))

        # Nút xuất file CSV để báo cáo
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Tải xuống báo cáo dự đoán (.csv)",
            data=csv_data,
            file_name="fraud_predictions_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi hệ thống khi xử lý: {e}")