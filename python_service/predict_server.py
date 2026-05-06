"""
predict_server.py
=================
Flask 预测服务，供 Spring Boot 后端调用。

启动方式：
    python predict_server.py

接口：
    POST /predict          使用 CNN+Transformer 模型预测
    POST /predict/lgbm     使用 LightGBM 模型预测
    POST /predict/lstm     使用 LSTM 模型预测
    GET  /health           健康检查
    GET  /models           查询已加载模型信息

请求体（JSON）：
    {
        "data": [
            {
                "time": "2024-01-01 00:00:00",
                "load_mw": 70000,
                "temperature_c": -5.0,
                "dewpoint_c": -8.0,
                "wind_speed_mps": 3.5,
                "relative_humidity": 72.0,
                "hour": 0,
                "weekday": 0,
                "month": 1,
                "day": 1,
                "is_weekend": 0
            },
            ... （至少24条，取最后24条作为输入窗口）
        ]
    }

返回体（JSON）：
    {
        "code": 200,
        "message": "success",
        "model": "CNN+Transformer",
        "input_hours": 24,
        "forecast_horizon": 24,
        "predictions": [
            {"step": 1, "load_mw": 71234.5, "mape_ref": 2.13},
            ...
        ]
    }
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# ---- 路径配置 ----
# ALGO_ROOT: Python 算法项目根目录（power_forecast 所在位置）
# 如果 predict_server.py 和 power_forecast 不在同一目录，在下面手动指定绝对路径
# 例如：ALGO_ROOT = Path(r"C:\Users\42998\Desktop\power_forecast")
ALGO_ROOT = Path(r"C:\Users\42998\Desktop\power_forecast")
PROJECT_ROOT = ALGO_ROOT
sys.path.insert(0, str(ALGO_ROOT))

from data_pipeline.feature_engineering import (
    add_wavelet_features,
    add_meteorological_features,
)
from models.cnn_transformer_model import CNNTransformerForecaster
from models.lgbm_model import LGBMForecastModel

app = Flask(__name__)

# ======================================================================
# 全局模型加载（启动时加载一次，避免每次请求重复加载）
# ======================================================================

MODELS = {}
SCALER = {}
FEATURE_COLS = []


def load_all_models():
    global MODELS, SCALER, FEATURE_COLS

    print("[Server] 加载 scaler 和特征列...")
    sc = np.load(
        str(PROJECT_ROOT / "data" / "sequences" / "scaler_stats.npz"),
        allow_pickle=True,
    )
    SCALER["means"] = sc["means"]
    SCALER["stds"]  = sc["stds"]
    SCALER["feature_cols"] = sc["feature_cols"].tolist()

    with open(PROJECT_ROOT / "data" / "sequences" / "feature_columns.json", encoding="utf-8") as f:
        FEATURE_COLS = json.load(f)

    print(f"[Server] 特征列({len(FEATURE_COLS)}个): {FEATURE_COLS}")

    # ---- CNN+Transformer ----
    import torch
    cnn_path = PROJECT_ROOT / "train" / "outputs" / "logs" / "cnn_transformer" / "best_model.pt"
    if cnn_path.exists():
        ckpt = torch.load(str(cnn_path), map_location="cpu")
        model = CNNTransformerForecaster(
            input_size=ckpt["input_size"],
            d_model=ckpt["d_model"],
            nhead=ckpt["nhead"],
            num_encoder_layers=ckpt["num_encoder_layers"],
            dim_feedforward=ckpt.get("dim_feedforward", 256),
            dropout=ckpt.get("dropout", 0.1),
            cnn_kernel_sizes=ckpt.get("cnn_kernel_sizes", (3, 5)),
            forecast_horizon=ckpt["forecast_horizon"],
            output_size=ckpt["output_size"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        MODELS["cnn_transformer"] = model
        print("[Server] CNN+Transformer 模型加载完成")
    else:
        print(f"[WARN] CNN+Transformer 模型文件不存在: {cnn_path}")

    # ---- LightGBM ----
    lgbm_path = PROJECT_ROOT / "train" / "outputs" / "logs" / "lgbm_baseline" / "best_lgbm_model.pkl"
    if lgbm_path.exists():
        MODELS["lgbm"] = LGBMForecastModel.load(str(lgbm_path))
        print("[Server] LightGBM 模型加载完成")
    else:
        print(f"[WARN] LightGBM 模型文件不存在: {lgbm_path}")

    # ---- LSTM ----
    import torch
    lstm_path = PROJECT_ROOT / "train" / "outputs" / "logs" / "lstm_baseline" / "best_lstm_model.pt"
    if lstm_path.exists():
        from models.lstm_model import LSTMRegressor
        ckpt = torch.load(str(lstm_path), map_location="cpu")
        lstm = LSTMRegressor(
            input_size=ckpt["input_size"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
            output_size=ckpt["output_size"],
            forecast_horizon=ckpt["forecast_horizon"],
        )
        lstm.load_state_dict(ckpt["model_state_dict"])
        lstm.eval()
        MODELS["lstm"] = lstm
        print("[Server] LSTM 模型加载完成")
    else:
        print(f"[WARN] LSTM 模型文件不存在: {lstm_path}")

    print(f"[Server] 共加载 {len(MODELS)} 个模型: {list(MODELS.keys())}")


# ======================================================================
# 数据预处理
# ======================================================================

def preprocess_input(data: list) -> np.ndarray:
    """
    将请求中的原始数据处理成模型输入序列。

    步骤：
    1. 转为 DataFrame，确保时间排序
    2. 补充衍生特征（小波分解 + 气象衍生）
    3. 标准化
    4. 取最后24行构造输入窗口
    5. 返回 [1, 24, F] 的 numpy 数组
    """
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 补全时间特征（如果前端没传）
    if "hour" not in df.columns:
        df["hour"] = df["time"].dt.hour
    if "weekday" not in df.columns:
        df["weekday"] = df["time"].dt.weekday
    if "month" not in df.columns:
        df["month"] = df["time"].dt.month
    if "day" not in df.columns:
        df["day"] = df["time"].dt.day
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # 添加衍生特征
    df = add_wavelet_features(df)
    df = add_meteorological_features(df)

    # 缺失值处理
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()

    # 检查特征列是否齐全
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")

    # 取最后24行
    if len(df) < 24:
        raise ValueError(f"输入数据至少需要24条，当前只有 {len(df)} 条")
    window = df[FEATURE_COLS].iloc[-24:].values.astype(np.float32)

    # 标准化
    means = SCALER["means"]
    stds  = SCALER["stds"]
    scaler_cols = SCALER["feature_cols"]
    for i, col in enumerate(FEATURE_COLS):
        if col in scaler_cols:
            idx = scaler_cols.index(col)
            window[:, i] = (window[:, i] - means[idx]) / stds[idx]

    return window[np.newaxis, :, :]   # [1, 24, F]


def inverse_load(scaled_arr: np.ndarray) -> np.ndarray:
    """将标准化的 load_mw 预测值还原为真实 MW"""
    scaler_cols = SCALER["feature_cols"]
    idx = scaler_cols.index("load_mw")
    return scaled_arr * SCALER["stds"][idx] + SCALER["means"][idx]


def format_predictions(pred_mw: np.ndarray, model_name: str, mape_ref: float) -> list:
    """格式化预测结果为 JSON 友好的列表"""
    pred_flat = pred_mw.reshape(-1)
    return [
        {
            "step": int(i + 1),
            "load_mw": round(float(pred_flat[i]), 2),
            "mape_ref": mape_ref,
        }
        for i in range(len(pred_flat))
    ]


# ======================================================================
# API 接口
# ======================================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "code": 200,
        "status": "ok",
        "loaded_models": list(MODELS.keys()),
    })


@app.route("/models", methods=["GET"])
def get_models():
    info = {}
    if "cnn_transformer" in MODELS:
        m = MODELS["cnn_transformer"]
        info["cnn_transformer"] = {
            "name": "CNN+Transformer",
            "parameters": m.count_parameters(),
            "forecast_horizon": m.forecast_horizon,
            "mape_ref": 2.13,
        }
    if "lgbm" in MODELS:
        info["lgbm"] = {
            "name": "LightGBM",
            "forecast_horizon": MODELS["lgbm"].forecast_horizon,
            "mape_ref": 1.75,
        }
    if "lstm" in MODELS:
        m = MODELS["lstm"]
        info["lstm"] = {
            "name": "LSTM",
            "forecast_horizon": m.forecast_horizon,
            "mape_ref": 2.09,
        }
    return jsonify({"code": 200, "models": info})


def _do_predict(model_key: str, model_display_name: str, mape_ref: float):
    """通用预测逻辑"""
    try:
        body = request.get_json()
        if not body or "data" not in body:
            return jsonify({"code": 400, "message": "请求体需包含 'data' 字段"}), 400

        if model_key not in MODELS:
            return jsonify({"code": 503, "message": f"{model_display_name} 模型未加载"}), 503

        # 预处理
        X = preprocess_input(body["data"])   # [1, 24, F]

        # 推理
        model = MODELS[model_key]

        if model_key in ("cnn_transformer", "lstm"):
            import torch
            with torch.no_grad():
                x_tensor = torch.tensor(X, dtype=torch.float32)
                pred_scaled = model(x_tensor).numpy()  # [1, H, 1] 或 [1, 1]
        else:
            pred_scaled = model.predict(X, FEATURE_COLS)  # LightGBM

        pred_mw = inverse_load(pred_scaled)
        predictions = format_predictions(pred_mw, model_display_name, mape_ref)

        return jsonify({
            "code": 200,
            "message": "success",
            "model": model_display_name,
            "input_hours": 24,
            "forecast_horizon": len(predictions),
            "predictions": predictions,
        })

    except ValueError as e:
        return jsonify({"code": 400, "message": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"code": 500, "message": f"服务器内部错误: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict_cnn():
    return _do_predict("cnn_transformer", "CNN+Transformer", 2.13)


@app.route("/predict/lgbm", methods=["POST"])
def predict_lgbm():
    return _do_predict("lgbm", "LightGBM", 1.75)


@app.route("/predict/lstm", methods=["POST"])
def predict_lstm():
    return _do_predict("lstm", "LSTM", 2.09)


# ======================================================================
# 启动
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Power Forecast Python Prediction Server")
    print("=" * 60)
    load_all_models()
    print("\n[Server] 启动在 http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)