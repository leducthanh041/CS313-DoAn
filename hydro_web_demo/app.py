
from flask import Flask, request, jsonify, render_template
import os, json
from pathlib import Path
from utils.inference import HydroDemoRuntime
from utils.eda_utils import (
    load_thuy_dien_cleaned,
    list_numeric_columns,
    build_univariate_payload,
    generate_univariate_pngs,
    generate_target_year_boxplots_pngs,
    generate_corr_heatmap_png,
    build_multivariate_stats,
)
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- config: paths ---
DEFAULT_CSV_PATHS = [
    Path("./data_thuydien/data_thuydien_cleaned.csv"),
    Path("./data/data_thuydien_cleaned.csv"),
    Path("./data_thuydien_cleaned.csv"),
]

def _resolve_csv_path():
    env_p = os.environ.get("TD_CLEANED_CSV", "").strip()
    if env_p and Path(env_p).exists():
        return Path(env_p)
    for p in DEFAULT_CSV_PATHS:
        if p.exists():
            return p
    return DEFAULT_CSV_PATHS[0]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

paths = {
    "enriched_csv": os.path.join(DATA_DIR, "data_thuydien_enriched.csv"),
    "xgb_model":    os.path.join(MODELS_DIR, "xgb_model.json"),
    "xgb_feats":    os.path.join(MODELS_DIR, "features_used_xgb.json"),
    "dtree_model":  os.path.join(MODELS_DIR, "dtree_model.pkl"),
    "dtree_feats":  os.path.join(MODELS_DIR, "features_used_dtree.json"),
    "cnn_model":    os.path.join(MODELS_DIR, "cnn_lstm_model_seq6_28d.pth"),
    "seq_scaler":   os.path.join(MODELS_DIR, "seq_scaler_28d_stride1_6ch.pkl"),
    "y_scaler":     os.path.join(MODELS_DIR, "y_scaler_28d_stride1_6ch.pkl"),
}

# If only one shared features_used.json is available, reuse for both
if not os.path.exists(paths["xgb_feats"]) and os.path.exists(os.path.join(MODELS_DIR, "features_used_xgb.json")):
    paths["xgb_feats"] = os.path.join(MODELS_DIR, "features_used_xgb.json")
if not os.path.exists(paths["dtree_feats"]) and os.path.exists(os.path.join(MODELS_DIR, "features_used_dtree.json")):
    paths["dtree_feats"] = os.path.join(MODELS_DIR, "features_used_dtree.json")

runtime = HydroDemoRuntime(paths)

@app.route("/index", methods=["GET"])
def index():
    info = {
        "enriched_range": [str(runtime.df_enriched.index.min()), str(runtime.df_enriched.index.max())],
        "models_loaded": {
            "xgb": runtime.xgb is not None,
            "dtree": runtime.dtree is not None,
            "cnn_lstm": runtime.cnn is not None
        }
    }
    return render_template("index.html", info=info)  # KHÔNG json.dumps(info)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    t_str = data.get("datetime")
    result = runtime.predict_all(t_str)
    return jsonify(result)

# --- route: /eda ---
@app.route("/eda", methods=["GET", "POST"])
def eda_home():

    csv_path = _resolve_csv_path()
    df, info = load_thuy_dien_cleaned(csv_path)

    # ── form params ─────────────────────────────────────────────
    numeric_cols = list_numeric_columns(df)
    target_default = (
        "muc_nuoc_thuong_luu_m"
        if "muc_nuoc_thuong_luu_m" in numeric_cols
        else (numeric_cols[0] if numeric_cols else None)
    )
    tab           = request.values.get("tab", "uni")  # 'uni'/'multi'
    uni_var       = request.values.get("uni_var", target_default)
    uni_resample  = request.values.get("uni_resample", "D")
    uni_bins      = int(request.values.get("uni_bins", 30))
    uni_by_month  = request.values.get("uni_by_month", "1") == "1"

    multi_cols    = request.values.getlist("multi_cols") or numeric_cols[:8]
    corr_method   = request.values.get("corr_method", "pearson")
    target        = request.values.get("target", target_default)
    lag_hours_max = int(request.values.get("lag_hours_max", 72))
    lag_step      = int(request.values.get("lag_step", 6))

    # ── payload số liệu (để render bảng thống kê, KHÔNG dùng Plotly) ──
    uni_payload = (
        build_univariate_payload(df, var=uni_var, resample=uni_resample, bins=uni_bins, by_month=uni_by_month)
        if uni_var else {"error": "No numeric column found."}
    )
    # multi_payload vẫn giữ nếu bạn dùng ở phần khác (bảng text); nếu chưa cần có thể set = {}
    multi_payload = {}

    # ── sinh PNG đơn biến (Seaborn/Matplotlib) ──────────────────
    try:
        uni_plot_files = generate_univariate_pngs(df, [uni_var]) if uni_var else []
    except Exception as e:
        print("generate_univariate_pngs error:", e)
        uni_plot_files = []
    
    # Lấy 1 file duy nhất (nếu có)
    uni_plot_file = uni_plot_files[0] if uni_plot_files else None

    # --- PNG: Boxplot theo tháng/giờ cho từng năm của biến TARGET ---
    try:
        target_box_files = generate_target_year_boxplots_pngs(df, target=target)
    except Exception as e:
        print("generate_target_year_boxplots_pngs error:", e)
        target_box_files = []

    # --- sinh PNG ĐA BIẾN (chỉ 1 ảnh heatmap theo multi_cols đã chọn) ---
    try:
        multi_corr_file = generate_corr_heatmap_png(df, multi_cols, method=corr_method)
    except Exception as e:
        print("generate_corr_heatmap_png error:", e)
        multi_corr_file = None
    
    # --- số liệu thống kê đa biến ---
    try:
        multi_stats = build_multivariate_stats(
            df=df,
            selected_cols=multi_cols,
            target=target,
            corr_method=corr_method,
            max_corr_matrix_cols=12,   # điều chỉnh nếu bạn muốn hiển thị nhiều/ít cột hơn
        )
    except Exception as e:
        print("build_multivariate_stats error:", e)
        multi_stats = {}

    print("multi_stats keys:", multi_stats.keys() if isinstance(multi_stats, dict) else type(multi_stats))


    

    # ── render ──────────────────────────────────────────────────
    return render_template(
        "eda.html",
        csv_path=str(csv_path.resolve()),
        info=info,
        tab=tab,
        numeric_cols=numeric_cols,
        uni_var=uni_var,
        uni_resample=uni_resample,
        uni_bins=uni_bins,
        uni_by_month=1 if uni_by_month else 0,
        multi_cols_selected=multi_cols,
        corr_method=corr_method,
        target=target,
        lag_hours_max=lag_hours_max,
        lag_step=lag_step,

        # bảng thống kê
        uni_payload=uni_payload,
        multi_payload=multi_payload,

        # chỉ dùng PNG
        uni_plot_file=uni_plot_file,
        target_box_files=target_box_files,
        multi_corr_file=multi_corr_file,
        multi_stats=multi_stats,
    )


@app.route("/", methods=["GET"]) 
def console_home():
    return render_template("console.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
