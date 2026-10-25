"""
Shared inference logic for the Flask edge service and the copilot's
predict_rul tool, so both validate input and score it the same way.
"""
import math
import os

import numpy as np
import pandas as pd

import data_prep as dp
from evaluate import load_model, MODEL_DIR

# Maintenance thresholds in cycles. With the 125-cycle cap these are a health
# index of 40% (WARNING) and 16% (CRITICAL); the manual uses the same numbers.
WARN_RUL, CRITICAL_RUL = 50, 20
MAX_CYCLES = 1000


class InputError(ValueError):
    """Bad request payload; the message is safe to return to the caller."""


def alert_for(rul):
    if rul <= CRITICAL_RUL:
        return "CRITICAL", "Schedule maintenance immediately: imminent failure risk."
    if rul <= WARN_RUL:
        return "WARNING", "Plan maintenance in an upcoming window to avoid unplanned downtime."
    return "OK", "Equipment healthy. Continue normal operation."


class RulPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.model, self.meta = load_model(model_dir)

    @property
    def required_sensors(self):
        return list(self.meta["sensors"])

    def _frame(self, cycles):
        if not isinstance(cycles, list) or not cycles:
            raise InputError("'cycles' must be a non-empty list of per-cycle readings")
        if len(cycles) > MAX_CYCLES:
            raise InputError(f"at most {MAX_CYCLES} cycles per request")
        min_cycles = self.meta.get("min_cycles", 1)
        if len(cycles) < min_cycles:
            raise InputError(f"send at least {min_cycles} consecutive cycles "
                             f"(got {len(cycles)}); the model uses {min_cycles}-cycle windows")
        if not all(isinstance(c, dict) for c in cycles):
            raise InputError("each item in 'cycles' must be an object of sensor readings")

        need = self.required_sensors
        missing = sorted({s for c in cycles for s in need if s not in c},
                         key=lambda s: int(s.split("_")[1]))
        if missing:
            raise InputError(f"missing sensor readings: {', '.join(missing)} "
                             f"(required: {', '.join(need)})")
        rows = []
        for i, c in enumerate(cycles):
            row = {}
            for s in need:
                v = c[s]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise InputError(f"cycle {i}: {s} must be a finite number")
                row[s] = float(v)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.insert(0, "cycle", np.arange(1, len(df) + 1))
        df.insert(0, "unit", 1)
        return df

    def predict(self, cycles, unit_id=1):
        """cycles: most recent readings, oldest first, each a dict that holds at
        least the sensors in meta['sensors']. Returns RUL, health index, alert."""
        df = self._frame(cycles)
        m = self.meta
        feats = dp.engineer(df, m["sensors"], tuple(m["windows"]), m["baseline"])
        x_last = feats.iloc[[-1]][m["feature_cols"]]
        rul = float(np.clip(self.model.predict(x_last)[0], 0, m["rul_clip"]))
        level, msg = alert_for(rul)
        return {
            "unit_id": unit_id,
            "predicted_rul_cycles": round(rul, 1),
            "health_index": round(100 * rul / m["rul_clip"], 1),
            "alert": level,
            "recommendation": msg,
        }


_DEFAULT = None


def get_predictor():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RulPredictor(os.environ.get("RUL_MODEL_DIR", MODEL_DIR))
    return _DEFAULT
