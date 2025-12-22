import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(page_title="Raw Data Viewer", layout="wide")

# -----------------------------
# Performance principles
# 1) Never auto-replot on every widget change -> use st.form + submit button
# 2) Cache: bytes -> DataFrame; DataFrame -> prepared (datetime index, sorted, numeric cast)
# 3) Reduce points before plotting: resample + (optional) decimate for very large series
# 4) Plotly Scattergl for fast rendering in browser (WebGL)
# -----------------------------

# ---------- Caching ----------
@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes, encoding: str) -> pd.DataFrame:
    # Use low_memory=False to avoid dtype thrashing; still fast enough with caching.
    return pd.read_csv(BytesIO(file_bytes), encoding=encoding, low_memory=False)

@st.cache_data(show_spinner=False)
def prepare_df(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col)
    d = d.set_index(time_col)

    # Best-effort numeric conversion for object columns (keeps non-numeric as object)
    for c in d.columns:
        if d[c].dtype == "object":
            d[c] = pd.to_numeric(d[c], errors="ignore")
    return d

@st.cache_data(show_spinner=False)
def resample_numeric(df: pd.DataFrame, freq: str, sum_cols: tuple) -> pd.DataFrame:
    if freq == "RAW":
        # Keep only numeric for plotting
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        return df[num_cols].copy()

    sum_cols = set(sum_cols)
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    d = df[num_cols].copy()
    if len(num_cols) == 0:
        return d

    agg = {c: ("sum" if c in sum_cols else "mean") for c in num_cols}
    out = d.resample(freq).agg(agg)
    out = out.dropna(how="all")
    return out

def decimate(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Uniformly subsample rows to at most max_points for client-side rendering speed."""
    if max_points is None or max_points <= 0:
        return df
    n = len(df)
    if n <= max_points:
        return df
    step = int(np.ceil(n / max_points))
    return df.iloc[::step].copy()

def detect_time_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "datetime", "time", "timestamp", "date", "Date", "DATE", "td", "ngay_gio", "thoi_gian"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Heuristic: first column that can be parsed with decent success
    for c in df.columns[:5]:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() >= 0.6:
            return c
    return None

def plot_series_webgl(dfi: pd.DataFrame, cols: list[str], title: str):
    fig = go.Figure()
    x = dfi.index
    for c in cols:
        if c not in dfi.columns:
            continue
        y = dfi[c]
        fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", name=c))
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- UI ----------------
st.title("Raw Data Viewer (Fast)")

with st.sidebar:
    st.subheader("Tải dữ liệu")
    mode = st.radio("Nguồn dữ liệu", ["Upload CSV"], index=0)
    encoding = st.selectbox("Encoding", ["utf-8-sig", "utf-8", "latin1"], index=0)
    data_kind = st.radio("Loại dữ liệu", ["Thủy điện", "Thời tiết"], index=0)

st.markdown(
    """
Trang này được tối ưu để:
- Không re-plot liên tục khi bạn đang chọn tuỳ chọn (chỉ plot khi bấm nút).
- Resample + giảm số điểm trước khi vẽ.
- Dùng Plotly WebGL (`Scattergl`) để render mượt khi dữ liệu lớn.
"""
)

uploaded = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded is None:
    st.info("Hãy upload một file CSV để bắt đầu.")
    st.stop()

# Read -> detect time column
raw_bytes = uploaded.getvalue()
df0 = read_csv_bytes(raw_bytes, encoding=encoding)

time_col = detect_time_column(df0)
if time_col is None:
    st.error("Không nhận diện được cột thời gian. Hãy đổi tên cột thời gian thành 'datetime' (khuyến nghị) hoặc chọn thủ công bên dưới.")
    time_col = st.selectbox("Chọn cột thời gian", options=list(df0.columns))

else:
    time_col = st.selectbox("Cột thời gian", options=list(df0.columns), index=list(df0.columns).index(time_col))

df = prepare_df(df0, time_col=time_col)

# Preview
with st.expander("Xem trước dữ liệu (50 dòng)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True, height=260)

# Plot controls: wrap in a form to avoid replot every change
with st.form("plot_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with c1:
        freq = st.selectbox("Resample", ["RAW", "1H", "3H", "6H", "12H", "1D"], index=3)
    with c2:
        max_points = st.selectbox("Giới hạn điểm vẽ", [2000, 5000, 10000, 20000, "Không giới hạn"], index=1)
        max_points_val = None if max_points == "Không giới hạn" else int(max_points)
    with c3:
        show_stats = st.checkbox("Hiện thống kê nhanh", value=True)
    with c4:
        plot_btn = st.form_submit_button("Vẽ biểu đồ")

# Quick stats without heavy ops
if show_stats:
    left, right = st.columns(2)
    with left:
        st.write(f"**Số dòng (sau chuẩn hoá):** {len(df):,}")
        st.write(f"**Khoảng thời gian:** {df.index.min()} → {df.index.max()}")
    with right:
        num_cols_all = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        st.write(f"**Số cột số:** {len(num_cols_all)}")
        st.write(f"**Số cột tổng:** {df.shape[1]}")

if not plot_btn:
    st.stop()

# Resample numeric + decimate
sum_cols = ("precipitation", "precip", "rain", "luong_mua", "precipitation (mm)")
dfn = resample_numeric(df, freq=freq, sum_cols=sum_cols)
dfn = decimate(dfn, max_points=max_points_val)

if dfn.shape[1] == 0:
    st.error("Không có cột số để vẽ (hoặc resample đã loại hết).")
    st.stop()

# Plot logic by kind
if data_kind == "Thủy điện":
    # Prefer canonical column names; fallback to numeric columns if missing
    preferred = ["muc_nuoc_thuong_luu_m", "luu_luong_den_ho_m3_s", "luu_luong_den_ho_m3_s_capped"]
    cols = [c for c in preferred if c in dfn.columns]

    if len(cols) == 0:
        # fallback: let user select
        st.warning("Không tìm thấy các cột chuẩn của thủy điện. Hãy chọn cột số để vẽ.")
        cols = st.multiselect("Chọn cột để vẽ", options=list(dfn.columns), default=list(dfn.columns)[:2])

    if len(cols) == 0:
        st.error("Bạn chưa chọn cột nào để vẽ.")
        st.stop()

    plot_series_webgl(dfn, cols, title="Thủy điện — Chuỗi theo thời gian")

else:
    # Weather: let user choose columns; provide Select all + sensible defaults
    all_cols = list(dfn.columns)

    # Heuristic defaults
    default_candidates = [
        "temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
        "wind_speed_10m", "wind_direction_10m", "surface_pressure"
    ]
    defaults = [c for c in default_candidates if c in all_cols]
    if len(defaults) == 0:
        defaults = all_cols[:6]

    top_bar = st.columns([1, 1, 2])
    with top_bar[0]:
        select_all = st.checkbox("Chọn tất cả cột", value=False)
    with top_bar[1]:
        per_chart = st.selectbox("Số cột / biểu đồ", [1, 2, 3, 4], index=1)
    with top_bar[2]:
        cols = st.multiselect("Chọn cột thời tiết để vẽ", options=all_cols, default=(all_cols if select_all else defaults))

    if len(cols) == 0:
        st.error("Bạn chưa chọn cột nào để vẽ.")
        st.stop()

    # Batch plotting for speed: split into groups of per_chart
    st.subheader("Thời tiết — Chuỗi theo thời gian")
    for i in range(0, len(cols), per_chart):
        grp = cols[i:i+per_chart]
        plot_series_webgl(dfn, grp, title=" | ".join(grp))