import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# 2. TẢI MÔ HÌNH VÀ BỘ CHUẨN HÓA (SCALER)
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


# 3. GIAO DIỆN CHÍNH & UPLOAD FILE
st.title("💳 Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng")

uploaded_file = st.file_uploader(
    "Tải lên tập tin .csv chứa dữ liệu giao dịch cần kiểm tra",
    type=["csv"]
)

# 4. ĐIỀU CHỈNH NGƯỠNG THRESHOLD
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

# 5. XỬ LÝ DỮ LIỆU & HIỂN THỊ KẾT QUẢ
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        results_df = df.copy()
        
        feature_cols = [col for col in df.columns if col not in ["Class", "Prediction"]]
        X = df[feature_cols]
        
        # --- CHUẨN HÓA (SCALE) DỮ LIỆU TỰ ĐỘNG ---
        try:
            # Nhận diện nếu file upload là file Raw chưa xử lý
            if "Time" in X.columns and "Amount" in X.columns:
                X['scaled_amount'] = scaler.transform(X['Amount'].values.reshape(-1, 1))
                X['scaled_time'] = scaler.transform(X['Time'].values.reshape(-1, 1))
                X.drop(['Time', 'Amount'], axis=1, inplace=True)
                
                # Sắp xếp lại đúng thứ tự cột để đưa vào mô hình 
                cols = ['scaled_amount', 'scaled_time'] + [c for c in X.columns if c not in ['scaled_amount', 'scaled_time']]
                X = X[cols]
                
            elif "scaled_time" in X.columns and "scaled_amount" in X.columns:
                # Nếu dữ liệu đã có sẵn cột chuẩn hóa, bỏ qua bước này
                pass 
        except Exception as scale_err:
            st.error(f"❌ Lỗi chuẩn hóa dữ liệu đầu vào: {scale_err}")
            st.stop()

        # --- CHẠY MÔ HÌNH DỰ ĐOÁN ---
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            results_df["Fraud_Probability"] = probabilities
        else:
            predictions = model.predict(X)
            results_df["Fraud_Probability"] = predictions.astype(float)
            st.warning("⚠️ Mô hình hiện tại không hỗ trợ trích xuất xác suất.")

        results_df["Prediction"] = predictions

        # --- Tính toán thống kê ---
        total_transactions = len(results_df)
        fraud_count = int(np.sum(predictions == 1))
        normal_count = total_transactions - fraud_count
        fraud_percentage = (fraud_count / total_transactions) * 100

        # --- HIỂN THỊ THỐNG KÊ TỔNG QUAN ---
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

        # Tách dataframe cho các mục đích hiển thị khác nhau
        fraud_df = results_df[results_df["Prediction"] == 1]
        normal_df = results_df[results_df["Prediction"] == 0]

        if len(fraud_df) > 0 and "Fraud_Probability" in fraud_df.columns:
            fraud_df = fraud_df.sort_values(by="Fraud_Probability", ascending=False)


        # 6. TRÌNH QUẢN LÝ GIAO DỊCH (TÌM KIẾM & LỌC)
        st.markdown("---")
        st.subheader("📋 Trình Quản Lý Giao Dịch")

        # Khung chứa bộ lọc và thanh tìm kiếm
        col_filter, col_search = st.columns([2, 1])
        
        with col_filter:
            view_mode = st.radio(
                "Bộ lọc danh sách:",
                options=["🚨 Giao dịch gian lận", "✅ Giao dịch bình thường", "🌐 Toàn bộ giao dịch"],
                horizontal=True
            )
            
        with col_search:
            search_query = st.text_input("🔍 Tìm kiếm theo ID:", placeholder="Nhập ID giao dịch...")

        # Áp dụng bộ lọc Radio Button
        if view_mode == "🚨 Chỉ giao dịch gian lận":
            display_df = fraud_df.copy()
        elif view_mode == "✅ Chỉ giao dịch bình thường":
            display_df = normal_df.copy()
        else:
            display_df = results_df.copy()

        # Áp dụng bộ lọc Thanh tìm kiếm
        if search_query:
            display_df = display_df[display_df.index.astype(str).str.contains(search_query.strip())]

        # Hiển thị kết quả ra màn hình
        if len(display_df) > 0:
            st.write(f"Đang hiển thị **{len(display_df):,}** giao dịch:")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("⚠️ Không tìm thấy giao dịch nào phù hợp với điều kiện lọc/tìm kiếm của bạn.")
            
        # 7. TRÍ TUỆ NHÂN TẠO GIẢI THÍCH (XAI - SHAP)
        st.markdown("---")
        st.subheader("🧠 Giải thích Giao Dịch bằng AI (XAI - SHAP)")
        st.info("Tại sao mô hình lại phân loại như vậy? Hãy chọn một ID Giao Dịch từ bảng phía trên để xem AI giải thích.")
        
        # Chỉ hiển thị công cụ phân tích nếu có dữ liệu trên bảng
        if len(display_df) > 0:
            col_sel, col_btn = st.columns([1, 2])
            with col_sel:
                selected_idx = st.selectbox("🔍 Chọn ID Giao Dịch:", display_df.index.tolist())
            
            with col_btn:
                st.write("") 
                st.write("")
                analyze_btn = st.button("Phân tích giao dịch này")

            if analyze_btn:
                with st.spinner("Đang trích xuất dữ liệu hộp đen bằng SHAP..."):
                    try:
                        # Rút trích đúng 1 dòng dữ liệu mà khách hàng chọn
                        instance_to_explain = X.loc[[selected_idx]]
                        
                        # Khởi tạo TreeExplainer cho mô hình ExtraTrees/XGBoost
                        explainer = shap.TreeExplainer(model)
                        shap_values_raw = explainer.shap_values(instance_to_explain)
                        
                        # Xử lý đồng bộ mảng dữ liệu SHAP tùy thuộc phiên bản thư viện
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
                        
                        # Đóng gói thành đối tượng shap
                        explanation = shap.Explanation(
                            values=shap_vals,
                            base_values=base_val,
                            data=instance_to_explain.iloc[0].values,
                            feature_names=instance_to_explain.columns.tolist()
                        )
                        
                        # Vẽ biểu đồ Waterfall
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(explanation, max_display=10, show=False)
                        st.pyplot(fig)
                        
                        st.success("✅ Trích xuất thành công! Các thanh màu **ĐỎ** (hướng sang phải) là yếu tố làm tăng khả năng gian lận. Các thanh màu **XANH** (hướng sang trái) là yếu tố chứng minh giao dịch an toàn.")
                    except Exception as e:
                        st.error(f"❌ Lỗi vẽ biểu đồ SHAP. Chi tiết: {e}")

        # 8. TẢI XUỐNG BÁO CÁO DỮ LIỆU
        st.markdown("---")
        st.subheader("📥 Xuất Báo Cáo Dữ Liệu")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            csv_all = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 Tải Bản Đầy Đủ", data=csv_all, file_name="all_transactions.csv", mime="text/csv", use_container_width=True)
        with dl_col2:
            csv_fraud = fraud_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="🚨 Tải DS Gian Lận", data=csv_fraud, file_name="fraud_only.csv", mime="text/csv", use_container_width=True, disabled=(len(fraud_df) == 0))
        with dl_col3:
            csv_normal = normal_df.to_csv(index=False).encode("utf-8")
            st.download_button(label="✅ Tải DS Bình Thường", data=csv_normal, file_name="normal_only.csv", mime="text/csv", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi hệ thống khi xử lý: {e}")
