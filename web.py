import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler 

# ----------------------------------------------------------------------
# PHẦN 1: ĐỊNH NGHĨA KIẾN TRÚC MODEL (Model 6-feature của bạn)
# ----------------------------------------------------------------------
# (Kiến trúc model của bạn giữ nguyên, không thay đổi)
class CNNLSTM(nn.Module):
    def __init__(self, seq_feat_dim,
                 conv_filters=16, lstm_hidden=64, dense_hidden=32, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(seq_feat_dim, conv_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 28 / 4 = 7. (Nếu L=28)
        # 672 / 4 = 168. (Nếu L=672) -> Kiến trúc vẫn hợp lệ
        self.lstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_hidden, num_layers=1, batch_first=True) 
        self.norm = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, dense_hidden), nn.ReLU(),
            nn.Linear(dense_hidden, 1)
        )

    def forward(self, x_seq):
        # x_seq: (N, L, F) -> (N, F, L)
        x = x_seq.transpose(1, 2)
        x = self.conv(x)        # (N, F', L')
        x = x.transpose(1, 2)   # (N, L', F')
        out, _ = self.lstm(x)     # (N, L', H)
        h = out[:, -1, :]       # (N, H)
        h = self.norm(h)
        h = self.dropout(h)
        return self.head(h).squeeze(1) 

# ----------------------------------------------------------------------
# PHẦN 2: CẤU HÌNH (Cập nhật theo file và yêu cầu)
# ----------------------------------------------------------------------

# --- Đường dẫn (File bạn đã tải lên) ---
MODEL_PATH = "cnn_lstm/cnn_lstm_model_seq6_28d.pth"
SCALER_X_PATH = "cnn_lstm/seq_scaler_28d_stride1_6ch.pkl"
SCALER_Y_PATH = "cnn_lstm/y_scaler_28d_stride1_6ch.pkl"

# --- Tham số (Từ file và yêu cầu của bạn) ---
# *** ĐÃ SỬA ***
# Lỗi "ValueError: X has 168 features, but StandardScaler is expecting 4032 features"
# 168 = 28 * 6
# 4032 = 672 * 6
# Điều này có nghĩa là scaler đã được train trên SEQ_LEN = 672, không phải 28.
SEQ_LEN = 672
SEQ_FEAT_DIM = 6

# --- Tên 6 feature mà model của bạn cần (từ code bạn dán) ---
# File .ipynb và scaler ngụ ý rằng model được train trên 6 cột DỮ LIỆU THÔ.
# Scaler (scaler_X) sẽ tự động chuẩn hóa 6 cột này.
# Tên cột phải khớp với file CSV đầu vào của bạn (ví dụ: data_thuydien_enriched.csv)
INPUT_FEATURE_NAMES = [
    'tong_luong_xa_qua_dap_tran_m3_s', # THAY VÌ 'dap_tran_norm'
    'tong_luong_xa_qua_nha_may_m3_s', # THAY VÌ 'nha_may_norm'
    'temperature_2m (°C)',
    'relative_humidity_2m (%)',
    'precipitation (mm)',
    'cloud_cover (%)'
]
TARGET_NAME = "Mực nước thượng lưu"
HOURS_PER_DAY = 24 # Giả định dữ liệu là hàng giờ

# ----------------------------------------------------------------------
# PHẦN 3: CÁC HÀM TẢI MODEL VÀ SCALERS
# ----------------------------------------------------------------------
# (Giữ nguyên, không thay đổi)
@st.cache_resource
def load_model(model_path, seq_feat_dim):
    """ Tải model CNNLSTM (6-feature) """
    try:
        # Khởi tạo đúng kiến trúc model với 6 features
        model = CNNLSTM(seq_feat_dim=seq_feat_dim) 
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        st.sidebar.success(f"✔️ Model '{Path(model_path).name}' đã tải.")
        return model
    except FileNotFoundError:
        st.sidebar.error(f"❌ Lỗi: Không tìm thấy file model tại '{model_path}'.")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải model: {e}")
        st.sidebar.info("Đảm bảo file .pth nằm trong thư mục gốc.")
        return None

@st.cache_resource
def load_scaler(scaler_path, scaler_name):
    """ Tải scaler (StandardScaler) từ file .pkl """
    try:
        scaler = joblib.load(scaler_path)
        # Kiểm tra nhanh số features của scaler
        if hasattr(scaler, 'n_features_in_'):
            st.sidebar.info(f"Scaler '{scaler_name}' mong đợi {scaler.n_features_in_} features.")
        return scaler
    except FileNotFoundError:
        st.sidebar.error(f"❌ Lỗi: Không tìm thấy file scaler tại '{scaler_path}'.")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải scaler '{scaler_name}': {e}")
        return None

# ----------------------------------------------------------------------
# PHẦN 4: HÀM DỰ BÁO LẶP
# ----------------------------------------------------------------------

# *** ĐÃ SỬA: Thêm tùy chọn chiến lược dự báo ***
def iterative_forecast(model, initial_seq_np, scaler_X, scaler_y, n_steps, strategy='repeat_last'):
    """
    Thực hiện dự báo lặp cho n_steps.
    
    strategy:
    - 'repeat_last': Giả định 6 feature ở bước cuối cùng (ví dụ: 12:00) 
                     sẽ không đổi cho 24 giờ tới.
    - 'repeat_cycle': Giả định 6 feature của 24 giờ qua (chu kỳ hàng ngày) 
                      sẽ lặp lại cho 24 giờ tới.
    """
    
    current_seq_np = initial_seq_np.copy() # Shape (672, 6) - Dữ liệu THÔ
    predictions = []
    
    # --- Chọn chiến lược cập nhật features ---
    future_features = []
    if strategy == 'repeat_last':
        # Chiến lược 1: Lấy 6 feature ở bước cuối cùng
        last_known_features = current_seq_np[-1, :].reshape(1, SEQ_FEAT_DIM) # Shape (1, 6)
        # Tạo một danh sách [last_row, last_row, ..., last_row] dài n_steps
        future_features = [last_known_features] * n_steps
        
    elif strategy == 'repeat_cycle':
        # Chiến lược 2: Lấy 24 feature của 24 giờ qua
        last_cycle_features = current_seq_np[-HOURS_PER_DAY:, :] # Shape (24, 6)
        # Lặp lại chu kỳ này cho n_steps
        for i in range(n_steps):
            # Lấy feature tương ứng trong chu kỳ 24h
            feature_for_this_step = last_cycle_features[i % HOURS_PER_DAY, :].reshape(1, SEQ_FEAT_DIM)
            future_features.append(feature_for_this_step)
            
    # Biến đổi scaler_X yêu cầu đầu vào shape (N, L*F)
    # L = 672, F = 6 -> L*F = 4032
    L = SEQ_LEN
    F = SEQ_FEAT_DIM
    
    with torch.no_grad():
        for i in range(n_steps):
            # 1. Scale chuỗi hiện tại (Dữ liệu THÔ)
            # Reshape (672, 6) -> (1, 672*6) hay (1, 4032)
            current_seq_2d = current_seq_np.reshape(1, L * F)
            
            # Scaler_X (từ file pkl) sẽ chuẩn hóa 4032 features thô này
            # Đây là dòng gây lỗi, bây giờ nó sẽ nhận đúng shape (1, 4032)
            scaled_seq_2d = scaler_X.transform(current_seq_2d)
            
            # 2. Reshape cho model (1, 4032) -> (1, 672, 6)
            scaled_seq_3d = scaled_seq_2d.reshape(1, L, F)
            input_tensor = torch.tensor(scaled_seq_3d, dtype=torch.float32)

            # 3. Dự báo
            pred_scaled = model(input_tensor) # Output shape (1,)
            
            # 4. Giải chuẩn hóa
            # Scaler_y yêu cầu input shape (N, 1)
            pred_scaled_2d = pred_scaled.reshape(-1, 1).cpu().numpy()
            pred_real = scaler_y.inverse_transform(pred_scaled_2d)
            
            prediction_value = pred_real.item()
            predictions.append(prediction_value)

            # 5. Cập nhật chuỗi (dữ liệu THÔ) cho bước tiếp theo
            # Lấy feature đã được chuẩn bị cho bước i
            next_feature_row = future_features[i]
            
            # Bỏ dòng đầu tiên (cũ nhất), thêm dòng feature mới (giả định) vào cuối
            current_seq_np = np.vstack((current_seq_np[1:], next_feature_row))
            
    return predictions

# ----------------------------------------------------------------------
# PHẦN 5: GIAO DIỆN STREAMLIT
# ----------------------------------------------------------------------
# (Các thay đổi về text và kiểm tra độ dài sẽ tự động cập nhật theo SEQ_LEN)
st.set_page_config(page_title=f"Dự báo {TARGET_NAME}", layout="centered")
st.title(f"🌊 Ứng dụng dự báo {TARGET_NAME}")
st.markdown("Dựa trên model `CNNLSTM` (6 features, seq_len 672)")
st.markdown("---")

# --- Sidebar để tải tài nguyên ---
st.sidebar.header("Tải tài nguyên")
model = load_model(MODEL_PATH, SEQ_FEAT_DIM)
scaler_X = load_scaler(SCALER_X_PATH, "Input (X)")
scaler_y = load_scaler(SCALER_Y_PATH, "Target (Y)")

# Chỉ tiếp tục nếu tất cả tải thành công
if not (model and scaler_X and scaler_y):
    st.error("⚠️ Một số tài nguyên cần thiết chưa được tải. Vui lòng kiểm tra sidebar.")
    st.info(f"Hãy đảm bảo các file `{MODEL_PATH}`, `{SCALER_X_PATH}`, và `{SCALER_Y_PATH}` nằm cùng thư mục với `app.py` này.")
    st.stop() # Dừng thực thi nếu thiếu file

# --- Layout chính ---
st.header("Cấu hình dự báo")

# 1. Tải file CSV
uploaded_file = st.file_uploader(
    f"Tải lên file CSV chứa dữ liệu quá khứ ({SEQ_LEN} giờ gần nhất)", 
    type=["csv"]
)

# 2. Nhập số ngày dự báo
n_days_forecast = st.number_input(
    "Nhập số ngày cần dự báo tới:", 
    min_value=1, max_value=7, value=3, step=1
)

# *** ĐÃ THÊM: Cho phép chọn chiến lược ***
forecast_strategy = st.selectbox(
    "Chọn chiến lược giả định cho 6 features (thời tiết, xả) trong tương lai:",
    options=['repeat_last', 'repeat_cycle'],
    index=0,
    format_func=lambda x: "Lặp lại giá trị cuối cùng" if x == 'repeat_last' else "Lặp lại chu kỳ 24 giờ qua"
)
strategy_info_text = {
    'repeat_last': f"Model sẽ dùng 6 feature cuối cùng trong file CSV và coi chúng **không đổi** cho tất cả {n_days_forecast * HOURS_PER_DAY} giờ dự báo.",
    'repeat_cycle': f"Model sẽ dùng 6 feature của **24 giờ cuối cùng** trong file CSV và giả định chu kỳ 24 giờ này **lặp lại** cho {n_days_forecast} ngày tới."
}


# 3. Nút bấm
if st.button(f"🚀 Dự báo cho {n_days_forecast} ngày tới"):
    if uploaded_file is None:
        st.warning("Vui lòng tải lên file CSV đầu vào.")
        st.stop()

    try:
        n_steps_forecast = n_days_forecast * HOURS_PER_DAY
        
        with st.spinner(f"Đang đọc file '{uploaded_file.name}'..."):
            df = pd.read_csv(uploaded_file)
            
            # Kiểm tra xem có đủ 6 cột cần thiết không
            missing_cols = [col for col in INPUT_FEATURE_NAMES if col not in df.columns]
            if missing_cols:
                st.error(f"❌ File CSV bị thiếu các cột cần thiết: {missing_cols}")
                st.info(f"File của bạn phải chứa 6 cột: {INPUT_FEATURE_NAMES}")
                st.stop()
            
            # Kiểm tra xem có đủ 672 dòng không
            if len(df) < SEQ_LEN:
                st.error(f"❌ File CSV cần ít nhất {SEQ_LEN} dòng dữ liệu. File của bạn chỉ có {len(df)} dòng.")
                st.stop()

            # Lấy 672 dòng cuối cùng và 6 cột cần thiết (dữ liệu THÔ)
            initial_seq_df = df[INPUT_FEATURE_NAMES].tail(SEQ_LEN)
            
            # Kiểm tra xem có dữ liệu rỗng (NaN) không
            if initial_seq_df.isnull().values.any():
                st.error("❌ Dữ liệu đầu vào (672 dòng cuối) chứa giá trị rỗng (NaN). Vui lòng làm sạch dữ liệu trước khi tải lên.")
                st.dataframe(initial_seq_df[initial_seq_df.isnull().any(axis=1)])
                st.stop()
                
            initial_seq_np = initial_seq_df.to_numpy(dtype=np.float32) # Shape (672, 6)

        with st.spinner(f"Đang thực hiện dự báo lặp cho {n_steps_forecast} giờ..."):
            # Cập nhật thông báo dựa trên chiến lược đã chọn
            st.info(f"💡 **Giả định:** {strategy_info_text[forecast_strategy]}")
            
            forecast_results = iterative_forecast(
                model,
                initial_seq_np,
                scaler_X,
                scaler_y,
                n_steps_forecast,
                strategy=forecast_strategy # Truyền chiến lược đã chọn
            )

        if forecast_results:
            st.success(f"✅ Dự báo hoàn tất cho {n_days_forecast} ngày ({n_steps_forecast} giờ).")
            
            # Xác định thời điểm bắt đầu dự báo
            time_col = None
            if 'time' in df.columns:
                time_col = 'time'
            elif 'thoi_diem' in df.columns: # Thêm trường hợp phổ biến
                time_col = 'thoi_diem'
            
            start_forecast_time = None
            if time_col:
                try:
                    # Lấy thời điểm cuối cùng từ file CSV và + 1 giờ
                    last_time_str = df[time_col].iloc[-1]
                    last_time = pd.to_datetime(last_time_str)
                    start_forecast_time = last_time + pd.Timedelta(hours=1)
                except Exception as e:
                    st.warning(f"Không thể đọc cột thời gian ('{time_col}'='{last_time_str}'). Lỗi: {e}. Sẽ dùng thời gian hiện tại.")
            
            if start_forecast_time is None:
                st.warning("Không tìm thấy cột 'time'/'thoi_diem' hoặc đọc lỗi. Sẽ dùng thời gian hiện tại làm mốc.")
                start_forecast_time = pd.Timestamp.now().floor('H') + pd.Timedelta(hours=1)


            # Tạo DataFrame kết quả
            forecast_times = pd.date_range(start=start_forecast_time, periods=n_steps_forecast, freq='h')
            
            results_df = pd.DataFrame({
                'Thời điểm dự báo': forecast_times,
                f'Dự báo {TARGET_NAME} (m)': forecast_results
            })
            results_df[f'Dự báo {TARGET_NAME} (m)'] = results_df[f'Dự báo {TARGET_NAME} (m)'].round(3)

            st.subheader("Kết quả dự báo:")
            st.dataframe(results_df, use_container_width=True, height=350)

            # Vẽ biểu đồ
            st.subheader("Biểu đồ dự báo:")
            st.line_chart(results_df.set_index('Thời điểm dự báo'))
            
    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi trong quá trình dự báo: {e}")
        st.exception(e) # In ra chi tiết lỗi để debug
        st.error(f"Kiểm tra lại: \n1. Các file scaler có đúng là StandardScaler và được train với shape (N, 4032) không? \n2. File CSV có đủ 672 dòng và 6 cột dữ liệu số không?")