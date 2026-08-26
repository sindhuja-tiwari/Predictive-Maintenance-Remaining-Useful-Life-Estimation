"""
Evaluate the trained model on the official C-MAPSS test split.
Prediction is made on the LAST recorded cycle of each test engine and compared
to the true RUL in RUL_FD001.txt.
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import data_prep as dp
from train import nasa_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, "rul_xgb.json"))

    test_raw = dp.load_raw("test")
    rul_truth = np.loadtxt(os.path.join(dp.DATA_DIR, "RUL_FD001.txt")).astype(int)

    feats = dp.engineer(test_raw, meta["sensors"], tuple(meta["windows"]))
    feats = pd.concat([feats, test_raw[["unit", "cycle"]].reset_index(drop=True)
                       .add_suffix("_idx")], axis=1)
    # take the last row per unit
    last_idx = test_raw.groupby("unit")["cycle"].idxmax()
    X_last = feats.loc[last_idx, meta["feature_cols"]]

    pred = np.clip(model.predict(X_last), 0, meta["rul_clip"])
    truth = np.clip(rul_truth, 0, meta["rul_clip"])
    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    score = nasa_score(truth, pred)
    print(f"[TEST]  engines={len(truth)}  RMSE={rmse:.2f} cycles  NASA score={score:.1f}")


if __name__ == "__main__":
    main()
