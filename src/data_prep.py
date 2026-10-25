"""
Load C-MAPSS, compute Remaining Useful Life (RUL) labels, and engineer
time-series degradation features (rolling stats to capture sensor drift).

Every feature is computed the same way in training and at inference time:
rolling windows only look backwards, and the "delta" features compare each
reading with a fixed fleet-wide healthy baseline learned from the training
data. Nothing depends on where a caller's window happens to start.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get(
    "RUL_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))

INDEX_COLS = ["unit", "cycle"]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + OP_COLS + SENSOR_COLS

RUL_CLIP = 125        # piecewise-linear RUL: "as-good-as-new" until wear starts
BASELINE_CYCLES = 20  # first N cycles of each training engine define "healthy"


def load_raw(split="train", subset="FD001", data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, f"{split}_{subset}.txt")
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)


def load_rul_truth(subset="FD001", data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, f"RUL_{subset}.txt")
    return np.atleast_1d(np.loadtxt(path)).astype(int)


def add_rul(df, rul_truth=None):
    """Training: RUL = last cycle of the unit - current cycle (clipped).
    Test: RUL_FD001.txt gives the RUL after the last recorded cycle of unit i
    (line i), so RUL at each row = that value + cycles left in the record."""
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    remaining = max_cycle - df["cycle"]
    if rul_truth is not None:
        final = dict(zip(range(1, len(rul_truth) + 1), rul_truth))
        remaining = remaining + df["unit"].map(final)
    df["RUL"] = remaining.clip(upper=RUL_CLIP)
    return df


def select_useful_sensors(train_df):
    """Drop sensors that carry no degradation signal. In FD001 these are the
    constant sensors 1, 5, 10, 16, 18, 19 and sensor 6, which only ever takes
    two values. This leaves the 14 sensors used in the C-MAPSS literature."""
    return [c for c in SENSOR_COLS
            if train_df[c].nunique() > 2 and train_df[c].std() > 1e-6]


def healthy_baseline(train_df, sensor_cols, n_cycles=BASELINE_CYCLES):
    """Mean reading of each sensor over the first n cycles of every training
    engine: what a healthy engine looks like across the fleet."""
    early = train_df[train_df["cycle"] <= n_cycles]
    return {c: float(early[c].mean()) for c in sensor_cols}


def engineer(df, sensor_cols, windows=(5, 15), baseline=None):
    """Per-unit rolling mean/std to expose the degradation trend, plus each
    reading's deviation from the fleet healthy baseline (cumulative drift).
    Grouped by unit so windows never leak across engines."""
    if baseline is None:
        raise ValueError("engineer() needs the healthy baseline (see healthy_baseline)")
    df = df.sort_values(INDEX_COLS).reset_index(drop=True)
    frames = [df[INDEX_COLS + list(sensor_cols)].copy()]
    g = df.groupby("unit")[list(sensor_cols)]
    for w in windows:
        rmean = g.rolling(w, min_periods=1).mean().reset_index(drop=True)
        rmean.columns = [f"{c}_mean{w}" for c in sensor_cols]
        rstd = g.rolling(w, min_periods=1).std().reset_index(drop=True).fillna(0)
        rstd.columns = [f"{c}_std{w}" for c in sensor_cols]
        frames.extend([rmean, rstd])
    base = pd.Series(baseline)[list(sensor_cols)]
    delta = df[list(sensor_cols)] - base
    delta.columns = [f"{c}_delta" for c in sensor_cols]
    frames.append(delta)
    return pd.concat(frames, axis=1)


def feature_columns(sensor_cols, windows=(5, 15)):
    cols = list(sensor_cols)
    for w in windows:
        cols += [f"{c}_mean{w}" for c in sensor_cols]
        cols += [f"{c}_std{w}" for c in sensor_cols]
    cols += [f"{c}_delta" for c in sensor_cols]
    return cols


def last_cycle_rows(df):
    """Index of the last recorded cycle of each unit (df must be sorted)."""
    return df.groupby("unit")["cycle"].idxmax().values
