import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

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
        model = joblib.load("fraud_model.pkl")
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
# 3. GIAO DIỆN CHÍNH & UPLOAD FILE
# ==============================================================================
st.title("💳 Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng")

uploaded_file = st.file_uploader(
    "Tải lên tập tin .csv chứa dữ liệu giao dịch cần kiểm tra",
    type=["csv"]
)

# ==============================================================================
# 4. KHU VỰC ĐIỀU CHỈNH NGƯỠNG (THRESHOLD)
# ==============================================================================
st.markdown("---")
st.subheader("⚙️ Cấu hình bộ lọc rủi ro")

threshold = st.slider(
    "Ngưỡng quyết định gian lận (Threshold)",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
)
st.markdown("---")

# ==============================================================================
# 5. XỬ LÝ DỮ LIỆU & HIỂN THỊ KẾT QUẢ
# ==============================================================================
if uploaded_file is not None:
    try:
        # Đọc dữ liệu đầu vào
        df = pd.read_csv(uploaded_file)
        results_df = df.copy()
        
        feature_cols = [col for col in df.columns if col not in ["Class", "Prediction"]]
        X = df[feature_cols]
        
        # ----------------------------------------------------------------------
        # CHUẨN HÓA (SCALE) DỮ LIỆU TỰ ĐỘNG
        # ----------------------------------------------------------------------
        try:
            # Nhận diện nếu file upload là file Raw chưa xử lý
            if "Time" in X.columns and "Amount" in X.columns:
                from sklearn.preprocessing import RobustScaler
                rob_scaler = RobustScaler()
                
                # Thực hiện scale ngay trên App
                X['scaled_amount'] = rob_scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
                X['scaled_time'] = rob_scaler.fit_transform(X['Time'].values.reshape(-1, 1))
                X.drop(['Time', 'Amount'], axis=1, inplace=True)
                
                # Sắp xếp lại đúng thứ tự cột để đưa vào mô hình học máy
                cols = ['scaled_amount', 'scaled_time'] + [c for c in X.columns if c not in ['scaled_amount', 'scaled_time']]
                X = X[cols]
                
            elif "scaled_time" in X.columns and "scaled_amount" in X.columns:
                # Nếu dữ liệu đã có sẵn cột chuẩn hóa, bỏ qua bước này
                pass 
        except Exception as scale_err:
            st.error(f"❌ Lỗi chuẩn hóa dữ liệu đầu vào: {scale_err}")
            st.stop()

        # ----------------------------------------------------------------------
        # CHẠY MÔ HÌNH DỰ ĐOÁN
        # ----------------------------------------------------------------------
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            results_df["Fraud_Probability"] = probabilities
        else:
            predictions = model.predict(X)
            results_df["Fraud_Probability"] = predictions.astype(float)
            st.warning("⚠️ Mô hình hiện tại không hỗ trợ trích xuất xác suất.")

        results_df["Prediction"] = predictions

        # Tính toán thống kê
        total_transactions = len(results_df)
        fraud_count = int(np.sum(predictions == 1))
        normal_count = total_transactions - fraud_count
        fraud_percentage = (fraud_count / total_transactions) * 100

        # ==============================================================================
        # HIỂN THỊ THỐNG KÊ
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

        fraud_df = results_df[results_df["Prediction"] == 1]
        normal_df = results_df[results_df["Prediction"] == 0]

        if len(fraud_df) > 0 and "Fraud_Probability" in fraud_df.columns:
            fraud_df = fraud_df.sort_values(by="Fraud_Probability", ascending=False)

        # ==============================================================================
        # BẢNG DANH SÁCH & TÍNH NĂNG SHAP EXPLAINER
        # ==============================================================================
        st.subheader("🚨 Danh Sách Giao Dịch Nghi Ngờ Gian Lận")
        if len(fraud_df) > 0:
            st.write(f"⚠️ Phát hiện **{len(fraud_df):,}** giao dịch vượt ngưỡng rủi ro `{threshold}`:")
            st.dataframe(fraud_df) 
            
            # --- XAI (EXPLAINABLE AI) ---
            st.markdown("---")
            st.subheader("🧠 Trí Tuệ Nhân Tạo Giải Thích (XAI - SHAP)")
            st.info("Tại sao mô hình lại cho rằng giao dịch này là gian lận? Hãy chọn một Mã Giao Dịch (Row Index) từ bảng trên để xem AI giải thích.")
            
            col_sel, col_btn = st.columns([1, 2])
            with col_sel:
                selected_idx = st.selectbox("🔍 Chọn ID Giao Dịch (Index):", fraud_df.index.tolist())
            
            with col_btn:
                st.write("") 
                st.write("")
                analyze_btn = st.button("Phân tích giao dịch này")

            if analyze_btn:
                with st.spinner("Đang trích xuất dữ liệu hộp đen bằng SHAP..."):
                    try:
                        # 1. Rút trích đúng 1 dòng dữ liệu (đã được Scale)
                        instance_to_explain = X.loc[[selected_idx]]
                        
                        # 2. Khởi tạo TreeExplainer cho mô hình ExtraTrees/XGBoost
                        explainer = shap.TreeExplainer(model)
                        shap_values_raw = explainer.shap_values(instance_to_explain)
                        
                        # 3. Xử lý đồng bộ mảng dữ liệu SHAP (Bắt lỗi định dạng của thuật toán)
                        if isinstance(shap_values_raw, list):
                            shap_vals = shap_values_raw[1][0] 
                            base_val = explainer.expected_value[1]
                        elif len(shap_values_raw.shape) == 3:
                            shap_vals = shap_values_raw[0, :, 1]
                            base_val = explainer.expected_value
                            if isinstance(base_val, (list, np.ndarray)):
                                base_val = base_val[-1]
                        else:
                            shap_vals = shap_values_raw[0]
                            base_val = explainer.expected_value
                            if isinstance(base_val, (list, np.ndarray)):
                                base_val = base_val[-1]
                        
                        # 4. Đóng gói thành đối tượng shap.Explanation để tương thích với hàm vẽ Waterfall
                        explanation = shap.Explanation(
                            values=shap_vals,
                            base_values=base_val,
                            data=instance_to_explain.iloc[0].values,
                            feature_names=instance_to_explain.columns.tolist()
                        )
                        
                        # 5. Vẽ biểu đồ Waterfall
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(explanation, max_display=10, show=False)
                        st.pyplot(fig)
                        
                        st.success("✅ Trích xuất thành công! Các thanh màu **ĐỎ** (hướng sang phải) đại diện cho những yếu tố khiến giao dịch bị nghi ngờ. Các thanh màu **XANH** (hướng sang trái) đại diện cho yếu tố an toàn.")
                    except Exception as e:
                        st.error(f"❌ Lỗi vẽ biểu đồ SHAP. Chi tiết: {e}")
            
        else:
            st.success("✅ Tuyệt vời! Không phát hiện giao dịch gian lận nào.")

        # ==============================================================================
        # KHU VỰC TẢI XUỐNG DỮ LIỆU
        # ==============================================================================
        st.markdown("---")
        st.subheader("📥 Xuất Báo Cáo Dữ Liệu")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            csv_all = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Tải xuống Bản Đầy Đủ", data=csv_all, file_name="all_transactions.csv", mime="text/csv", use_container_width=True)
        with dl_col2:
            csv_fraud = fraud_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="🚨 Tải xuống DS Gian Lận", data=csv_fraud, file_name="fraud_only.csv", mime="text/csv", use_container_width=True, disabled=(len(fraud_df) == 0))
        with dl_col3:
            csv_normal = normal_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="✅ Tải xuống DS Bình Thường", data=csv_normal, file_name="normal_only.csv", mime="text/csv", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi hệ thống khi xử lý: {e}")
