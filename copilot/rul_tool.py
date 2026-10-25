"""
Wraps the trained C-MAPSS RUL model so the agent can call it as a tool.

Uses the same predictor as the Flask /predict endpoint (src/inference.py), so
input validation, features and alert thresholds are identical in both places.
"""
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from inference import InputError, get_predictor  # noqa: E402


def predict_rul(cycles, unit_id=1):
    """cycles: list of per-cycle dicts holding the model's required sensors
    (see required_sensors()). Returns RUL, health index, alert, recommendation,
    or {"error": ...} if the input is unusable."""
    try:
        return get_predictor().predict(cycles, unit_id=unit_id)
    except InputError as e:
        return {"error": str(e)}


def required_sensors():
    return get_predictor().required_sensors


# JSON schema describing this tool for the LLM (OpenAI/Anthropic tool-calling format).
RUL_TOOL_SCHEMA = {
    "name": "predict_rul",
    "description": (
        "Estimate Remaining Useful Life (RUL) in operating cycles for a turbofan "
        "engine from a window of recent sensor readings. Returns RUL, a 0-100 health "
        "index, and a maintenance alert (OK/WARNING/CRITICAL). Use when the user asks "
        "how much life is left, whether an engine needs maintenance, or to assess "
        "health from sensor data. Needs at least 15 consecutive cycles, each with "
        "sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, "
        "sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21. "
        "If the user has not supplied readings, ask for them instead of inventing values."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "unit_id": {"type": "integer", "description": "Engine identifier."},
            "cycles": {
                "type": "array",
                "description": "Recent cycles, oldest->newest. Each item maps sensor names to float readings.",
                "items": {"type": "object"},
            },
        },
        "required": ["cycles"],
    },
}
