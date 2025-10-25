
import json
import numpy as np
import pandas as pd
import torch
import joblib
from typing import Dict, Any

from .features import load_enriched_csv, build_feature_row_tree, build_seq6_for_cnn, infer_expect_L_from_scaler
from .models import load_xgb_model, load_dtree_pipeline, load_cnn_lstm

def robust_load_scaler(path_pkl: str):
    import pickle, gzip, bz2, lzma, pathlib
    p = pathlib.Path(path_pkl)
    if not p.exists():
        return None, f"Không thấy file: {p}"
    try:
        return joblib.load(p), f"Loaded by joblib.load({p})"
    except Exception as e:
        msg1 = f"joblib.load fail: {repr(e)}"
    data = p.read_bytes()
    try:
        return pickle.loads(data), f"Loaded by pickle.loads({p})"
    except Exception as e:
        msg2 = f"pickle.loads fail: {repr(e)}"
    try:
        return pickle.loads(gzip.decompress(data)), f"Loaded by gzip+pickle ({p})"
    except Exception as e:
        msg3 = f"gzip+pickle fail: {repr(e)}"
    try:
        return pickle.loads(bz2.decompress(data)), f"Loaded by bz2+pickle ({p})"
    except Exception as e:
        msg4 = f"bz2+pickle fail: {repr(e)}"
    try:
        return pickle.loads(lzma.decompress(data)), f"Loaded by lzma+pickle ({p})"
    except Exception as e:
        msg5 = f"lzma+pickle fail: {repr(e)}"
    return None, " | ".join([msg1,msg2,msg3,msg4,msg5])

class HydroDemoRuntime:
    def __init__(self, paths: Dict[str,str]):
        self.paths = paths
        self.df_enriched = load_enriched_csv(paths["enriched_csv"])

        cols = self.df_enriched.columns
        self.target_col = "muc_nuoc_thuong_luu_m"
        if self.target_col not in cols:
            raise ValueError("Thiếu cột 'muc_nuoc_thuong_luu_m' trong enriched.")
        self.inflow_col = "luu_luong_den_ho_m3_s_capped" if "luu_luong_den_ho_m3_s_capped" in cols else (
                          "luu_luong_den_ho_m3_s" if "luu_luong_den_ho_m3_s" in cols else None)
        if self.inflow_col is None:
            raise ValueError("Thiếu cột lưu lượng đến hồ trong enriched.")
        self.w_cols = [c for c in ["temp_c","rh_pct","precip_mm","cloud_pct"] if c in cols]

        self.xgb = None
        self.dtree = None
        self.cnn = None
        self.device = "cpu"

        self.features_xgb = None
        self.features_dtree = None

        try:
            self.xgb = load_xgb_model(paths["xgb_model"])
            with open(paths["xgb_feats"], "r", encoding="utf-8") as f:
                self.features_xgb = json.load(f)
        except Exception as e:
            self.xgb = None
            self.xgb_err = str(e)

        try:
            self.dtree = load_dtree_pipeline(paths["dtree_model"])
            with open(paths["dtree_feats"], "r", encoding="utf-8") as f:
                self.features_dtree = json.load(f)
        except Exception as e:
            self.dtree = None
            self.dtree_err = str(e)

        try:
            self.cnn = load_cnn_lstm(paths["cnn_model"], device=self.device)
            self.seq_scaler, self.seq_msg = robust_load_scaler(paths["seq_scaler"])
            self.y_scaler,   self.y_msg   = robust_load_scaler(paths["y_scaler"])
        except Exception as e:
            self.cnn = None
            self.cnn_err = str(e)

    def predict_all(self, t_str: str) -> Dict[str, Any]:
        t = pd.to_datetime(t_str, dayfirst=True, errors="coerce")
        if pd.isna(t):
            t = pd.to_datetime(t_str, errors="coerce")
        if pd.isna(t):
            return {"ok": False, "error": f"Không parse được thời gian: {t_str}"}

        result = {"ok": True, "t": t.isoformat(), "warnings": []}

        if self.xgb is not None and self.features_xgb is not None:
            row_xgb, t_anchor_xgb = build_feature_row_tree(self.df_enriched, t, 
                                                           [self.target_col, self.inflow_col] + self.w_cols,
                                                           self.features_xgb)
            y_xgb = float(self.xgb.predict(row_xgb)[0])
            result["xgb"] = {"y_pred": y_xgb, "t_anchor": pd.to_datetime(t_anchor_xgb).isoformat()}
        else:
            result["xgb"] = {"error": getattr(self, "xgb_err", "Không tải được XGB")}

        if self.dtree is not None and self.features_dtree is not None:
            row_dt, t_anchor_dt = build_feature_row_tree(self.df_enriched, t, 
                                                         [self.target_col, self.inflow_col] + self.w_cols,
                                                         self.features_dtree)
            y_dt = float(self.dtree.predict(row_dt)[0])
            result["dtree"] = {"y_pred": y_dt, "t_anchor": pd.to_datetime(t_anchor_dt).isoformat()}
        else:
            result["dtree"] = {"error": getattr(self, "dtree_err", "Không tải được Decision Tree")}

        if self.cnn is not None and (self.seq_scaler is not None) and (self.y_scaler is not None):
            try:
                expect_L = infer_expect_L_from_scaler(self.seq_scaler)
                x_seq_s, t_anchor_cnn, step_minutes, L = build_seq6_for_cnn(self.df_enriched, t,
                                                                             self.target_col, self.inflow_col, self.w_cols,
                                                                             self.seq_scaler, expect_L)
                with torch.no_grad():
                    xb = torch.tensor(x_seq_s, dtype=torch.float32, device=self.device)
                    yhat_s = self.cnn(xb).cpu().numpy().ravel()[0]
                y_pred = float(self.y_scaler.inverse_transform([[yhat_s]]).ravel()[0])
                result["cnn_lstm"] = {
                    "y_pred": y_pred,
                    "t_anchor": pd.to_datetime(t_anchor_cnn).isoformat(),
                    "L": int(L),
                    "step_minutes": int(step_minutes)
                }
            except Exception as e:
                result["cnn_lstm"] = {"error": str(e)}
        else:
            msg = "Thiếu model/scaler cho CNN+LSTM."
            if getattr(self, "seq_msg", None): msg += f" seq_scaler: {self.seq_msg}."
            if getattr(self, "y_msg", None):   msg += f" y_scaler: {self.y_msg}."
            result["cnn_lstm"] = {"error": msg}

        return result
