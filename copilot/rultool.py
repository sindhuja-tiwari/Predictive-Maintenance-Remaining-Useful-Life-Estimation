"""
Wraps the trained C-MAPSS RUL model so the agent can call it as a tool.

Keeps a single loaded model in memory and exposes predict_rul(), which accepts a
recent sensor window (list of per-cycle dicts) and returns RUL + a maintenance
alert — the same contract the Flask /predict endpoint uses, reused here.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

import sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
import data_prep as dp  # noqa: E402

MODEL_DIR = os.path.join(_HERE, "..", "models")
WARN_RUL, CRITICAL_RUL = 50, 20

_MODEL = None
_META = None


def _load():
    global _MODEL, _META
    if _MODEL is None:
        with open(os.path.join(MODEL_DIR, "meta.json")) as f:
            _META = json.load(f)
        _MODEL = xgb.XGBRegressor()
        _MODEL.load_model(os.path.join(MODEL_DIR, "rul_xgb.json"))
    return _MODEL, _META


def _alert(rul):
    if rul <= CRITICAL_RUL:
        return "CRITICAL", "Schedule maintenance immediately — imminent failure risk."
    if rul <= WARN_RUL:
        return "WARNING", "Plan maintenance in an upcoming window to avoid unplanned downtime."
    return "OK", "Equipment healthy. Continue normal operation."


def predict_rul(cycles, unit_id=1):
    """cycles: list of dicts with sensor_1..21 (+ optional op_setting_1..3).
    Returns dict: predicted_rul_cycles, health_index, alert, recommendation."""
    model, meta = _load()
    if not cycles:
        return {"error": "no cycles provided"}
    df = pd.DataFrame(cycles)
    df.insert(0, "cycle", np.arange(1, len(df) + 1))
    df.insert(0, "unit", unit_id)
    for c in dp.SENSOR_COLS + dp.OP_COLS:
        if c not in df.columns:
            df[c] = 0.0
    feats = dp.engineer(df, meta["sensors"], tuple(meta["windows"]))
    x_last = feats.iloc[[-1]][meta["feature_cols"]]
    rul = float(np.clip(model.predict(x_last)[0], 0, meta["rul_clip"]))
    level, msg = _alert(rul)
    return {
        "unit_id": unit_id,
        "predicted_rul_cycles": round(rul, 1),
        "health_index": round(100 * rul / meta["rul_clip"], 1),
        "alert": level,
        "recommendation": msg,
    }


# JSON schema describing this tool for the LLM (OpenAI/Anthropic tool-calling format).
RUL_TOOL_SCHEMA = {
    "name": "predict_rul",
    "description": (
        "Estimate Remaining Useful Life (RUL) in operating cycles for a turbofan "
        "engine from a window of recent sensor readings. Returns RUL, a 0-100 health "
        "index, and a maintenance alert (OK/WARNING/CRITICAL). Use when the user asks "
        "how much life is left, whether an engine needs maintenance, or to assess "
        "health from sensor data."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "unit_id": {"type": "integer", "description": "Engine identifier."},
            "cycles": {
                "type": "array",
                "description": "Recent cycles, oldest->newest. Each item maps sensor_1..sensor_21 (and optional op_setting_1..3) to float readings.",
                "items": {"type": "object"},
            },
        },
        "required": ["cycles"],
    },
}