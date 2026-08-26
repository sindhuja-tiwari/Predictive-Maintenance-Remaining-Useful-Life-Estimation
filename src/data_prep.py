"""
Load C-MAPSS, compute Remaining Useful Life (RUL) labels, and engineer
time-series degradation features (rolling stats to capture sensor drift).
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

INDEX_COLS = ["unit", "cycle"]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OP_COLS + SENSOR_COLS

# Sensors that are flat/constant in FD001 carry no degradation signal; drop them.
# Determined empirically (near-zero variance) — recomputed at load time to be safe.
RUL_CLIP = 125  # piecewise-linear RUL: health is "as-good-as-new" until wear starts


def load_raw(split="train", subset="FD001"):
    path = os.path.join(DATA_DIR, f"{split}_{subset}.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)
    return df


def add_rul(df, rul_truth=None):
    """Training: RUL = max cycle per unit - current cycle (clipped).
    Test: last cycle's RUL comes from RUL_FD001.txt, back-filled per unit."""
    if rul_truth is None:
        max_cycle = df.groupby("unit")["cycle"].transform("max")
        df["RUL"] = (max_cycle - df["cycle"]).clip(upper=RUL_CLIP)
    else:
        max_cycle = df.groupby("unit")["cycle"].transform("max")
        # remaining life at each row = (final RUL for unit) + (cycles left in record)
        final_rul = df["unit"].map(dict(zip(range(1, len(rul_truth) + 1), rul_truth)))
        df["RUL"] = (final_rul + (max_cycle - df["cycle"])).clip(upper=RUL_CLIP)
    return df


def select_useful_sensors(train_df):
    variances = train_df[SENSOR_COLS].var()
    useful = [c for c in SENSOR_COLS if variances[c] > 1e-6]
    return useful


def engineer(df, sensor_cols, windows=(5, 15)):
    """Per-unit rolling mean/std to expose degradation trend, not just instantaneous
    noise. Grouped by unit so windows never leak across engines."""
    df = df.sort_values(INDEX_COLS).reset_index(drop=True)
    feat_frames = [df[INDEX_COLS + sensor_cols].copy()]
    g = df.groupby("unit")[sensor_cols]
    for w in windows:
        rmean = g.rolling(w, min_periods=1).mean().reset_index(drop=True)
        rmean.columns = [f"{c}_mean{w}" for c in sensor_cols]
        rstd = g.rolling(w, min_periods=1).std().reset_index(drop=True).fillna(0)
        rstd.columns = [f"{c}_std{w}" for c in sensor_cols]
        feat_frames.extend([rmean, rstd])
    feats = pd.concat(feat_frames, axis=1)
    # difference from the unit's initial reading = cumulative drift
    first = df.groupby("unit")[sensor_cols].transform("first")
    delta = (df[sensor_cols] - first)
    delta.columns = [f"{c}_delta" for c in sensor_cols]
    feats = pd.concat([feats.reset_index(drop=True), delta.reset_index(drop=True)], axis=1)
    return feats


def feature_columns(sensor_cols, windows=(5, 15)):
    cols = list(sensor_cols)
    for w in windows:
        cols += [f"{c}_mean{w}" for c in sensor_cols]
        cols += [f"{c}_std{w}" for c in sensor_cols]
    cols += [f"{c}_delta" for c in sensor_cols]
    return cols
