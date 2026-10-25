"""
Train a Remaining Useful Life (RUL) regressor on C-MAPSS FD001.

Model: gradient-boosted trees (XGBoost). A strong tabular baseline that trains
in seconds and exports to a small artifact suited to edge inference.

Metrics:
  RMSE            standard regression error (cycles)
  NASA score      asymmetric penalty: late predictions (over-estimating RUL, i.e.
                  predicting failure later than reality) cost more than early
                  ones, matching the real cost of unplanned downtime.

The validation engines are held out by unit and used for early stopping. They
are also the engines the dashboard demo replays, so the demo only ever shows
engines the model never trained on.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

import data_prep as dp
from metrics import rmse, nasa_score
from evaluate import evaluate_test, MODEL_DIR

WINDOWS = (5, 15)
SEED = 42


def build_dataset(data_dir=None):
    train_raw = dp.load_raw("train", data_dir=data_dir)
    sensors = dp.select_useful_sensors(train_raw)
    baseline = dp.healthy_baseline(train_raw, sensors)
    train_raw = dp.add_rul(train_raw).sort_values(dp.INDEX_COLS).reset_index(drop=True)
    X = dp.engineer(train_raw, sensors, WINDOWS, baseline)
    feat_cols = dp.feature_columns(sensors, WINDOWS)
    df = pd.concat([X[feat_cols], train_raw[["unit", "cycle", "RUL"]]], axis=1)
    return df, feat_cols, sensors, baseline


def train(data_dir=None, model_dir=MODEL_DIR, n_estimators=2000, verbose=True):
    os.makedirs(model_dir, exist_ok=True)
    df, feat_cols, sensors, baseline = build_dataset(data_dir)

    # Split by unit so no engine appears in both train and validation.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(gss.split(df, groups=df["unit"]))
    tr, va = df.iloc[tr_idx], df.iloc[va_idx]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_lambda=2.0, n_jobs=1, random_state=SEED, tree_method="hist",
        early_stopping_rounds=100,
    )
    model.fit(tr[feat_cols], tr["RUL"],
              eval_set=[(va[feat_cols], va["RUL"])], verbose=False)

    pred = np.clip(model.predict(va[feat_cols]), 0, dp.RUL_CLIP)
    val_rmse = rmse(va["RUL"], pred)
    val_nasa_mean = nasa_score(va["RUL"].values, pred) / len(va)

    meta = {
        "dataset": "FD001",
        "feature_cols": feat_cols, "sensors": sensors, "baseline": baseline,
        "windows": list(WINDOWS), "min_cycles": max(WINDOWS),
        "rul_clip": dp.RUL_CLIP, "train_mean_rul": float(tr["RUL"].mean()),
        "best_iteration": int(model.best_iteration),
        "val_units": sorted(int(u) for u in va["unit"].unique()),
        "val_rmse": val_rmse, "val_nasa_score_per_row": val_nasa_mean,
    }
    test_res, _, _ = evaluate_test(model, meta, data_dir)
    meta.update(test_res)

    model.save_model(os.path.join(model_dir, "rul_xgb.json"))
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[validation]  RMSE={val_rmse:.2f} cycles  (all rows of "
              f"{len(meta['val_units'])} held-out engines, best_iteration={meta['best_iteration']})")
        print(f"[test]        RMSE={test_res['test_rmse']:.2f} cycles  "
              f"NASA score={test_res['test_nasa_score']:.1f}  (last cycle of 100 engines)")
        print(f"Saved model + meta to {os.path.abspath(model_dir)}")
    return model, meta


if __name__ == "__main__":
    train()
