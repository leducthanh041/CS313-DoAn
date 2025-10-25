
from flask import Flask, request, jsonify, render_template
import os, json
from utils.inference import HydroDemoRuntime

app = Flask(__name__)

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

@app.route("/", methods=["GET"])
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
