"""
Train a Remaining Useful Life (RUL) regressor on C-MAPSS.

Model: gradient-boosted trees (XGBoost). Strong tabular baseline that trains in
seconds and exports to a tiny artifact — ideal for the edge-inference bonus.

Metrics:
  RMSE            standard regression error (cycles)
  NASA score      asymmetric penalty: late predictions (over-estimating RUL, i.e.
                  predicting failure later than reality) are punished harder than
                  early ones, matching the real cost of unplanned downtime.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

import data_prep as dp

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
WINDOWS = (5, 15)


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def build_dataset():
    train_raw = dp.load_raw("train")
    sensors = dp.select_useful_sensors(train_raw)
    train_raw = dp.add_rul(train_raw)
    X = dp.engineer(train_raw, sensors, WINDOWS)
    feat_cols = dp.feature_columns(sensors, WINDOWS)
    df = pd.concat([X[feat_cols], train_raw[["unit", "RUL"]].reset_index(drop=True)], axis=1)
    return df, feat_cols, sensors


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df, feat_cols, sensors = build_dataset()

    # Split by unit so no engine appears in both train and validation.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(gss.split(df, groups=df["unit"]))
    tr, va = df.iloc[tr_idx], df.iloc[va_idx]

    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_lambda=2.0, n_jobs=4, random_state=42,
    )
    model.fit(tr[feat_cols], tr["RUL"],
              eval_set=[(va[feat_cols], va["RUL"])], verbose=False)

    pred = np.clip(model.predict(va[feat_cols]), 0, dp.RUL_CLIP)
    rmse = float(np.sqrt(mean_squared_error(va["RUL"], pred)))
    score = nasa_score(va["RUL"].values, pred)
    print(f"[validation]  RMSE={rmse:.2f} cycles   NASA score={score:.1f}")

    model.save_model(os.path.join(MODEL_DIR, "rul_xgb.json"))
    meta = {"feature_cols": feat_cols, "sensors": sensors,
            "windows": list(WINDOWS), "rul_clip": dp.RUL_CLIP,
            "val_rmse": rmse, "val_nasa_score": score}
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model + meta to {MODEL_DIR}")


if __name__ == "__main__":
    main()
