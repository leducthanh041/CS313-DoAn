
import re
import numpy as np
import pandas as pd
from collections import Counter

def normalize_col(c: str) -> str:
    c0 = re.sub(r"\s+", " ", str(c).strip().lower())
    c0 = re.sub(r"\s*\([^)]*\)", "", c0)
    c0 = c0.replace("%", "pct").replace("°", "")
    c0 = re.sub(r"[^a-z0-9_ ]+", "_", c0).replace(" ", "_")
    c0 = re.sub(r"_+", "_", c0).strip("_")
    return c0

def parse_best_datetime(series):
    s1 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    s2 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return s1 if s1.notna().sum() >= s2.notna().sum() else s2

def to_numeric_safe(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def load_enriched_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    time_candidates = [c for c in ["thoi_diem","time","datetime","timestamp"] if c in df.columns]
    if not time_candidates:
        raise ValueError("Thiếu cột thời gian trong enriched CSV.")
    time_col = time_candidates[0]
    dt = parse_best_datetime(df[time_col])
    df = df.assign(thoi_diem=dt).dropna(subset=["thoi_diem"]).sort_values("thoi_diem").reset_index(drop=True)
    if time_col != "thoi_diem":
        df = df.drop(columns=[time_col])

    rename_map = {}
    if "temperature_2m" in df.columns:       rename_map["temperature_2m"] = "temp_c"
    if "relative_humidity_2m" in df.columns: rename_map["relative_humidity_2m"] = "rh_pct"
    if "precipitation" in df.columns:        rename_map["precipitation"] = "precip_mm"
    if "cloud_cover" in df.columns:          rename_map["cloud_cover"] = "cloud_pct"
    df = df.rename(columns=rename_map)

    for c in df.columns:
        if c == "thoi_diem": continue
        df[c] = to_numeric_safe(df[c])

    return df.set_index("thoi_diem").sort_index()

ROLL_WINDOWS = ["24H","3D","7D","14D","28D"]
LAGS = [1,3,6,12,24,48,72]

def build_feature_row_tree(frame: pd.DataFrame, t, vars_used: list, feat_list: list):
    t = pd.to_datetime(t)
    idx = frame.index[frame.index < t]
    if len(idx) == 0:
        raise ValueError("Enriched không có quan sát nào trước thời điểm yêu cầu.")
    t_anchor = idx.max()

    sub = frame.loc[:t_anchor].iloc[:-1]

    out = {}
    out["hour"] = t.hour
    out["dow"] = t.dayofweek
    out["month"] = t.month
    out["is_weekend"] = int(out["dow"] >= 5)

    def roll_stats(s: pd.Series, prefix: str):
        r24  = s.rolling("24H", closed="left").agg(["mean","std","max","min"])
        r3d  = s.rolling("3D",  closed="left").agg(["mean","std","max","min"])
        r7d  = s.rolling("7D",  closed="left").agg(["mean","std","max","min"])
        r14d = s.rolling("14D", closed="left").agg(["mean","std","max","min"])
        r28d = s.rolling("28D", closed="left").agg(["mean","std","max","min"])
        last = {}
        for name, robj in [("24h", r24), ("3d", r3d), ("7d", r7d), ("14d", r14d), ("28d", r28d)]:
            last[f"{prefix}__mean_{name}"] = robj["mean"].iloc[-1] if len(robj) else np.nan
            last[f"{prefix}__std_{name}"]  = robj["std"].iloc[-1]  if len(robj) else np.nan
            last[f"{prefix}__max_{name}"]  = robj["max"].iloc[-1]  if len(robj) else np.nan
            last[f"{prefix}__min_{name}"]  = robj["min"].iloc[-1]  if len(robj) else np.nan
        return last

    def add_lags(s: pd.Series, prefix: str, lags=LAGS):
        for k in lags:
            out[f"{prefix}__lag_{k}"] = s.shift(k).iloc[-1] if len(s) >= k+1 else np.nan

    for col in vars_used:
        if col not in frame.columns:
            for w in ["24h","3d","7d","14d","28d"]:
                out[f"{col}__mean_{w}"] = np.nan
                out[f"{col}__std_{w}"]  = np.nan
                out[f"{col}__max_{w}"]  = np.nan
                out[f"{col}__min_{w}"]  = np.nan
            for k in LAGS:
                out[f"{col}__lag_{k}"] = np.nan
            continue
        s = frame[col].astype(float)
        s.index = pd.to_datetime(frame.index)
        s = s.loc[:t_anchor].iloc[:-1]
        out.update(roll_stats(s, col))
        add_lags(s, col)

    row = pd.DataFrame({k: [v] for k,v in out.items()})
    for c in feat_list:
        if c not in row.columns:
            row[c] = np.nan
    row = row[feat_list]
    return row, t_anchor

def infer_expect_L_from_scaler(scaler):
    n_features_in = getattr(scaler, "n_features_in_", None)
    if n_features_in and n_features_in % 6 == 0:
        return n_features_in // 6
    return None

def build_seq6_for_cnn(frame: pd.DataFrame, t, target_col: str, inflow_col: str, w_cols: list,
                        seq_scaler, expect_L=None):
    t = pd.to_datetime(t)
    idx = frame.index[frame.index < t]
    if len(idx) == 0:
        raise ValueError("Enriched không có quan sát nào trước thời điểm yêu cầu.")
    t_anchor = idx.max()

    vars_req = [target_col, inflow_col] + w_cols
    for col in vars_req:
        if col not in frame.columns:
            raise ValueError(f"Thiếu cột '{col}' trong enriched CSV.")

    EXPECT_L = expect_L if expect_L is not None else infer_expect_L_from_scaler(seq_scaler)
    total_minutes = 28*24*60
    if EXPECT_L is None:
        dmins = np.diff(frame.index.view("i8")) / 60_000_000_000
        dmins = dmins[dmins > 0]
        if len(dmins) > 0:
            from collections import Counter
            step_minutes = int(Counter(dmins.astype(int)).most_common(1)[0][0])
            EXPECT_L = max(1, int(round(total_minutes / step_minutes)))
        else:
            step_minutes = 60
            EXPECT_L = 28*24
    else:
        step_minutes = int(round(total_minutes / EXPECT_L))

    start_dt = t_anchor - pd.Timedelta(days=28)
    end_dt   = t_anchor - pd.Timedelta(minutes=step_minutes)
    grid = pd.date_range(start=start_dt, end=end_dt, freq=f"{step_minutes}min", inclusive="both")
    if len(grid) != EXPECT_L:
        grid = pd.date_range(start=start_dt, periods=EXPECT_L, freq=f"{step_minutes}min")

    base = frame[vars_req].sort_index()
    joined = pd.merge_asof(pd.DataFrame({"__dt": grid}),
                           base.reset_index().rename(columns={"thoi_diem":"__dt"}).sort_values("__dt"),
                           on="__dt", direction="backward").set_index("__dt").sort_index()

    joined[vars_req] = joined[vars_req].ffill().interpolate(limit_direction="both")
    if joined[vars_req].isna().any().any():
        missing_cols = joined.columns[joined.isna().any()].tolist()
        raise ValueError(f"Còn NaN sau fill/interpolate ở các cột: {missing_cols}")

    lvl = joined[target_col].values.astype("float32")
    inf = joined[inflow_col].values.astype("float32")
    w4  = joined[w_cols].values.astype("float32")
    seq6 = np.column_stack([lvl, inf, w4])
    if seq6.shape != (EXPECT_L, 6):
        raise ValueError(f"seq6 shape {seq6.shape} != ({EXPECT_L},6)")

    x2d = seq6.reshape(1, EXPECT_L*6).astype("float32")
    x2d_s = seq_scaler.transform(x2d).astype("float32")
    x_seq_s = x2d_s.reshape(1, EXPECT_L, 6)
    return x_seq_s, t_anchor, step_minutes, EXPECT_L
