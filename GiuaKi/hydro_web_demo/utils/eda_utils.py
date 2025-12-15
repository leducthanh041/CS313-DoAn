# =============================
# File: utils/eda_utils.py
# =============================
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import matplotlib
# Nếu chạy trên server không có display:
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib


TIME_COL_CANDIDATES = [
    "thoi_diem", "Thoi_diem", "Thời điểm", "thoi_diem_gmt7", "time", "datetime"
]

import seaborn as sns
import matplotlib.pyplot as plt

def build_multivariate_stats(
    df: pd.DataFrame,
    selected_cols: list[str],
    target: str,
    corr_method: str = "pearson",
    max_corr_matrix_cols: int = 12,
):
    """
    Trả về dict số liệu cho tab Đa biến:
      - n_rows, n_cols, columns
      - missing: list[{col, missing, missing_pct}]
      - corr_with_target: list[{var, corr, abs_corr}] (|corr| giảm dần)
      - corr_matrix: {columns: [...], rows: [[...], ...]}  (giới hạn số cột)
      - target_by_month_per_year: list[{year, rows:[{month,count,median,q25,q75,lb,ub}...]}]
    """
    result = {
        "n_rows": 0, "n_cols": 0, "columns": [],
        "missing": [], "corr_with_target": [],
        "corr_matrix": None,
        "target_by_month_per_year": []   # <— mới
    }

    df_b = df.copy()

    # --- Chuẩn hoá thời gian + tạo cột month/year ---
    dt = None
    if "thoi_diem" in df_b.columns:
        dt = pd.to_datetime(df_b["thoi_diem"], errors="coerce")
    elif isinstance(df_b.index, pd.DatetimeIndex):
        dt = pd.to_datetime(df_b.index, errors="coerce")

    if dt is not None:
        if isinstance(dt, pd.Series):
            df_b["month"] = dt.dt.month
            df_b["year"]  = dt.dt.year
        else:  # DatetimeIndex
            df_b["month"] = dt.month
            df_b["year"]  = dt.year

    # --- Cột được chọn ---
    cols = [c for c in selected_cols if c in df_b.columns]
    result["columns"] = cols
    result["n_cols"] = len(cols)
    if len(cols) == 0:
        return result

    # --- Numeric subset + missing ---
    df_num = df_b[cols].apply(pd.to_numeric, errors="coerce")
    n_rows = df_num.shape[0]
    result["n_rows"] = int(n_rows)

    for c in cols:
        miss = int(df_num[c].isna().sum())
        pct = (miss / n_rows * 100.0) if n_rows > 0 else 0.0
        result["missing"].append({"col": c, "missing": miss, "missing_pct": round(pct, 2)})

    # --- Corr matrix (giới hạn số cột để render) ---
    cm_cols = cols[:max_corr_matrix_cols]
    if len(cm_cols) >= 2:
        corr = df_num[cm_cols].corr(method=corr_method).round(3)
        result["corr_matrix"] = {
            "columns": list(corr.columns),
            "rows": corr.values.tolist(),
        }

    # --- Corr với target ---
    t_series = pd.to_numeric(df_b.get(target, pd.Series(index=df_b.index)), errors="coerce")
    df_ct = pd.concat([df_num, t_series.rename("__t__")], axis=1)
    corr_ct = df_ct.corr(method=corr_method)
    if "__t__" in corr_ct.columns:
        s = corr_ct["__t__"].drop(labels=["__t__"], errors="ignore").dropna()
        pairs = []
        for var, val in s.items():
            pairs.append({"var": var, "corr": float(round(val, 3)), "abs_corr": float(round(abs(val), 3))})
        pairs.sort(key=lambda x: x["abs_corr"], reverse=True)
        result["corr_with_target"] = pairs

    # --- Tóm tắt target theo THÁNG cho TỪNG NĂM ---
    if "month" in df_b.columns and "year" in df_b.columns:
        tg = pd.DataFrame({
            "target": t_series,
            "month": pd.to_numeric(df_b["month"], errors="coerce"),
            "year":  pd.to_numeric(df_b["year"],  errors="coerce"),
        }).dropna(subset=["target", "month", "year"])

        if not tg.empty:
            per_year = []
            for y, g_year in tg.groupby("year", dropna=True):
                rows = []
                for m, g in g_year.groupby("month", dropna=True)["target"]:
                    if g.empty:
                        continue
                    q25 = g.quantile(0.25)
                    med = g.median()
                    q75 = g.quantile(0.75)
                    iqr = q75 - q25
                    rows.append({
                        "month": int(m),
                        "count": int(g.count()),
                        "median": float(round(med, 3)),
                        "q25": float(round(q25, 3)),
                        "q75": float(round(q75, 3)),
                        "lb": float(round(q25 - 1.5*iqr, 3)),
                        "ub": float(round(q75 + 1.5*iqr, 3)),
                    })
                rows.sort(key=lambda r: r["month"])
                per_year.append({"year": int(y), "rows": rows})
            per_year.sort(key=lambda d: d["year"])
            result["target_by_month_per_year"] = per_year

    return result

def _safe_name(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("%", "pct")
         .replace("(", "")
         .replace(")", "")
         .strip()
    )

def generate_corr_heatmap_png(df, cols, method="pearson", out_dir="./static/eda_plots"):
    """
    Vẽ 1 correlation heatmap cho đúng danh sách biến 'cols' đã chọn.
    Trả về basename file PNG hoặc None nếu không đủ cột numeric.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Giữ đúng thứ tự biến người dùng chọn, đồng thời chỉ lấy cột có trong df
    cols = [c for c in cols if c in df.columns]

    # Lọc numeric
    df_num = df[cols].apply(pd.to_numeric, errors="coerce")
    # Cần >=2 cột numeric để có ma trận tương quan
    numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
    if len(numeric_cols) < 2:
        return None

    corr = df_num[numeric_cols].corr(method=method)

    # Đặt tên file ngắn gọn + hash để tránh quá dài
    key = f"{method}-" + "-".join([_safe_name(c) for c in numeric_cols])
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    fn = f"corr_{method}_{digest}.png"
    out_path = os.path.join(out_dir, fn)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(f"Tương quan ({method})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return fn

def generate_target_year_boxplots_pngs(
    df,
    target: str,
    out_dir: str = "./static/eda_plots",
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> list[str]:
    """
    Vẽ 1 ảnh/năm cho biến target:
      - Trên: Boxplot theo THÁNG
      - Dưới: Boxplot theo GIỜ
    Trục Y cố định theo [q_low, q_high] để so sánh giữa các năm (fallback min/max nếu quantile không hợp lệ).
    Trả về danh sách basename file PNG theo thứ tự năm tăng dần.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    df_b = df.copy()

    # Bảo đảm cột thời gian và cột target tồn tại
    if "thoi_diem" not in df_b.columns:
        raise ValueError("DataFrame thiếu cột 'thoi_diem'.")
    if target not in df_b.columns:
        return []

    # Chuẩn hoá thời gian & cột phụ trợ
    df_b["thoi_diem"] = pd.to_datetime(df_b["thoi_diem"], errors="coerce")
    df_b = df_b.dropna(subset=["thoi_diem"])
    df_b["year"] = df_b["thoi_diem"].dt.year
    df_b["month"] = df_b["thoi_diem"].dt.month
    df_b["hour"] = df_b["thoi_diem"].dt.hour

    # Ép numeric cho target để vẽ
    df_b[target] = pd.to_numeric(df_b[target], errors="coerce")

    if df_b[target].dropna().empty:
        return []

    # Trục Y cố định theo quantile (fallback min/max nếu cần)
    ymin = float(df_b[target].quantile(q_low))
    ymax = float(df_b[target].quantile(q_high))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin >= ymax:
        ymin, ymax = float(df_b[target].min()), float(df_b[target].max())

    years = sorted(y for y in df_b["year"].dropna().unique())
    order_month = list(range(1, 12 + 1))
    order_hour = list(range(0, 24))

    safe_t = _safe_name(target)

    for y in years:
        sub = df_b[df_b["year"] == y].copy()
        sub = sub.dropna(subset=[target, "month", "hour"])
        if sub.empty:
            continue

        # Nếu file đã tồn tại, bỏ qua render lại (tiết kiệm thời gian)
        fn = f"{safe_t}_box_month_hour_{int(y)}.png"
        out_path = os.path.join(out_dir, fn)
        if os.path.exists(out_path):
            saved.append(fn)
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

        # Boxplot theo THÁNG (hàng trên)
        sns.boxplot(x="month", y=target, data=sub, order=order_month, ax=axes[0])
        axes[0].set_title(f"{target} — Boxplot theo THÁNG (năm {int(y)})")
        axes[0].set_xlabel("Tháng")
        axes[0].set_ylabel(target)
        axes[0].set_ylim(ymin, ymax)

        # Boxplot theo GIỜ (hàng dưới)
        sns.boxplot(x="hour", y=target, data=sub, order=order_hour, ax=axes[1])
        axes[1].set_title(f"{target} — Boxplot theo GIỜ (năm {int(y)})")
        axes[1].set_xlabel("Giờ")
        axes[1].set_ylabel(target)
        axes[1].set_ylim(ymin, ymax)

        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(fn)

    return saved

def generate_univariate_pngs(df: pd.DataFrame, cols: list[str], out_dir: str = None) -> list[str]:
    """
    Sinh ảnh Boxplot + Histogram+KDE cho từng cột trong `cols`.
    Lưu ảnh vào static/eda_plots/ và trả về danh sách tên file (basename).
    """
    if out_dir is None:
        out_dir = os.path.join("static", "eda_plots")
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna().astype(float, errors="ignore")
        # nếu convert lỗi hoặc empty
        try:
            s = s.dropna().astype(float)
        except Exception:
            # nếu không thể cast sang float, bỏ qua
            continue
        if s.empty:
            continue

        # filename safe (replace spaces/đặc ký tự)
        safe_name = c.replace(" ", "_").replace("/", "_").replace("%", "pct")
        fn = f"{safe_name}_box_hist.png"
        out_path = os.path.join(out_dir, fn)

        # Tạo figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        # Boxplot
        try:
            sns.boxplot(x=s, ax=axes[0])
            axes[0].set_title(f"Boxplot — {c}")
            axes[0].set_xlabel(c)
        except Exception:
            axes[0].text(0.5, 0.5, "Không vẽ được boxplot", ha="center", va="center")

        # Histogram + KDE
        try:
            sns.histplot(s, bins=40, kde=True, ax=axes[1])
            axes[1].set_title(f"Histogram + KDE — {c}")
            axes[1].set_xlabel(c)
            axes[1].set_ylabel("Count")
        except Exception:
            axes[1].text(0.5, 0.5, "Không vẽ được histogram", ha="center", va="center")

        # Lưu file PNG
        try:
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
        finally:
            plt.close(fig)

        saved.append(fn)
    return saved

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase + normalize simple ascii; replace spaces/%/() and units
    cols = []
    for c in df.columns:
        cc = str(c).strip().lower()
        cc = cc.replace("(°c)", "").replace("(mm)", "").replace("(%)", "pct")
        cc = cc.replace("°c", "").replace("%", "pct").replace("/h", "_per_h")
        cc = cc.replace("(", "").replace(")", "").replace(" ", "_")
        cols.append(cc)
    df.columns = cols
    return df

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # Find time column and convert to datetime; name it 'thoi_diem'
    time_col = None
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        # Try heuristic: first datetime-like column
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="raise")
                time_col = c
                df[c] = parsed
                break
            except Exception:
                continue
    if time_col is None:
        raise ValueError("Không tìm thấy cột thời gian.")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    if time_col != "thoi_diem":
        df = df.rename(columns={time_col: "thoi_diem"})
    return df

def load_thuy_dien_cleaned(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _standardize_columns(df)
    df = _ensure_datetime(df)
    info = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "start": df["thoi_diem"].min(),
        "end": df["thoi_diem"].max(),
        "freq_hint_hours": _infer_median_step_hours(df["thoi_diem"]) or None,
    }
    return df, info

def _infer_median_step_hours(td: pd.Series) -> float | None:
    try:
        med = (td.sort_values().diff().dt.total_seconds().median())
        return round((med or 0) / 3600.0, 2) if pd.notnull(med) and med > 0 else None
    except Exception:
        return None

def list_numeric_columns(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep common interest columns first if present
    priority = ["muc_nuoc_thuong_luu_m", "luu_luong_den_ho_m3_s", "luu_luong_den_ho_m3_s_capped",
                "temperature_2m", "temp_c", "relative_humidity_2m", "rh_pct",
                "precipitation", "precip_mm", "cloud_cover", "cloud_pct"]
    ordered = [c for c in priority if c in num_cols] + [c for c in num_cols if c not in priority]
    return ordered

# ---------- Univariate ----------

def _monthly_outlier_bounds(s: pd.Series) -> pd.DataFrame:
    # Compute per-month IQR bounds
    dfm = s.to_frame("val")
    dfm["month"] = dfm.index.month
    agg = dfm.groupby("month")["val"].agg(["count", "median", "quantile"])  # placeholder
    # manual quartiles to avoid deprecated behavior
    q = dfm.groupby("month")["val"].quantile([0.25, 0.75]).unstack()
    q.columns = ["q1", "q3"]
    merged = dfm.groupby("month")["val"].agg(["count", "median"]).join(q)
    merged["iqr"] = merged["q3"] - merged["q1"]
    merged["lb"] = merged["q1"] - 1.5 * merged["iqr"]
    merged["ub"] = merged["q3"] + 1.5 * merged["iqr"]
    return merged

def build_univariate_payload(df: pd.DataFrame, var: str, resample: str = "D", bins: int = 30, by_month: bool = True) -> dict:
    if var not in df.columns:
        return {"error": f"Không tìm thấy biến {var}"}

    d = df.set_index("thoi_diem").sort_index()
    s = d[var].dropna()

    # Summary stats
    summary = {
        "count": int(s.shape[0]),
        "missing": int(d.shape[0] - s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }

    # Time trend (resampled)
    ts = s.resample(resample).median().dropna()
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name=f"{var} ({resample})"))
    fig_ts.update_layout(title=f"Xu hướng thời gian ({var}, resample={resample})",
                         margin=dict(l=10, r=10, t=40, b=10), height=320)

    # Histogram
    fig_hist = go.Figure(data=[go.Histogram(x=s.values, nbinsx=bins, name=var)])
    fig_hist.update_layout(title=f"Phân phối {var} (bins={bins})",
                           margin=dict(l=10, r=10, t=40, b=10), height=320)

    fig_month = None
    month_stats = None
    if by_month:
        month_series = s.copy()
        month_series.index = month_series.index.to_period("M").to_timestamp()
        df_box = pd.DataFrame({var: month_series.values, "month": month_series.index.month})
        fig_month = go.Figure()
        fig_month.add_trace(go.Box(y=df_box[var], x=df_box["month"], boxpoints="outliers", name=var))
        fig_month.update_layout(title=f"Phân phối theo tháng (boxplot) — {var}", xaxis_title="Tháng",
                                margin=dict(l=10, r=10, t=40, b=10), height=320)
        month_stats = _monthly_outlier_bounds(s)
        month_stats = month_stats.reset_index().rename(columns={"median": "month_median"})

    return {
        "summary": summary,
        "fig_ts": fig_ts.to_plotly_json(),
        "fig_hist": fig_hist.to_plotly_json(),
        "fig_month": fig_month.to_plotly_json() if fig_month else None,
        "month_stats": month_stats.to_dict(orient="records") if month_stats is not None else None,
    }

# ---------- Multivariate ----------

def _corr(df: pd.DataFrame, cols: list[str], method: str) -> pd.DataFrame:
    ok = [c for c in cols if c in df.columns]
    return df[ok].corr(method=method)

def _crosscorr(a: pd.Series, b: pd.Series, lags: list[int]) -> pd.DataFrame:
    # Compute correlation at positive lags: corr(a.shift(lag), b)
    out = []
    for L in lags:
        corr = a.shift(L).corr(b)
        out.append({"lag": int(L), "corr": float(corr) if pd.notnull(corr) else np.nan})
    return pd.DataFrame(out)

def build_multivariate_payload(
    df: pd.DataFrame,
    cols: list[str],
    corr_method: str = "pearson",
    target: str | None = None,
    lag_hours_max: int = 72,
    lag_step: int = 6,
) -> dict:
    d = df.copy()
    d = d.set_index("thoi_diem").sort_index()

    # Correlation matrix
    corr_df = _corr(d, cols, corr_method)
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        zmin=-1, zmax=1,
        colorbar=dict(title=f"{corr_method.title()} r")
    ))
    fig_heat.update_layout(title=f"Ma trận tương quan ({corr_method})",
                           margin=dict(l=60, r=20, t=40, b=60), height=420)

    # Scatter pairs for top correlated with target
    fig_scatter = None
    top_pairs = None
    if target and target in d.columns:
        others = [c for c in cols if c != target and c in d.columns]
        if others:
            corrs = d[others].corrwith(d[target]).abs().sort_values(ascending=False)
            top = corrs.head(3).index.tolist()
            fig_scatter = make_subplots(rows=len(top), cols=1, shared_xaxes=False, subplot_titles=[f"{t} vs {target}" for t in top])
            for i, c in enumerate(top, start=1):
                fig_scatter.add_trace(go.Scatter(x=d[c], y=d[target], mode="markers", name=f"{c} vs {target}", opacity=0.5), row=i, col=1)
            fig_scatter.update_layout(height=320*len(top), title="Scatter các biến liên quan nhất với đích", margin=dict(l=40, r=20, t=40, b=40))
            top_pairs = [{"var": c, "abs_corr": float(corrs[c])} for c in top]

    # Lag correlation (hours) for target vs each var
    fig_lag = None
    lag_table = None
    if target and target in d.columns:
        # infer step hours
        step_h = _infer_median_step_hours(d.index.to_series()) or 1.0
        steps = int(max(1, round(lag_step / step_h)))
        max_steps = int(max(1, round(lag_hours_max / step_h)))
        lag_list = list(range(0, max_steps + 1, steps))
        lag_df_all = []
        for c in cols:
            if c == target or c not in d.columns:
                continue
            cc = _crosscorr(d[c], d[target], lag_list)
            cc["var"] = c
            lag_df_all.append(cc)
        if lag_df_all:
            lag_df = pd.concat(lag_df_all, ignore_index=True)
            fig_lag = go.Figure()
            for v, sub in lag_df.groupby("var"):
                fig_lag.add_trace(go.Scatter(x=sub["lag"], y=sub["corr"], mode="lines+markers", name=v))
            fig_lag.update_layout(title=f"Tương quan theo độ trễ với {target} (bước giờ ~{step_h})",
                                  xaxis_title="Độ trễ (bước)", yaxis_title="corr",
                                  margin=dict(l=40, r=20, t=40, b=40), height=380)
            lag_table = lag_df.sort_values(["var", "lag"])[:5000].to_dict(orient="records")

    return {
        "corr_heat": fig_heat.to_plotly_json(),
        "scatter_top": fig_scatter.to_plotly_json() if fig_scatter else None,
        "top_pairs": top_pairs,
        "lag_corr": fig_lag.to_plotly_json() if fig_lag else None,
        "lag_table": lag_table,
    }