"""
Evaluate the trained model on the official C-MAPSS test split.

The prediction is made on the LAST recorded cycle of each test engine and
compared with the true RUL in RUL_FD001.txt (clipped at the same 125-cycle cap
used for training, the usual convention for FD001).
"""
import os
import json
import numpy as np
import xgboost as xgb

import data_prep as dp
from metrics import rmse, nasa_score

MODEL_DIR = os.environ.get(
    "RUL_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))


def evaluate_test(model, meta, data_dir=None):
    test_raw = dp.load_raw("test", data_dir=data_dir)
    rul_truth = dp.load_rul_truth(data_dir=data_dir)

    feats = dp.engineer(test_raw, meta["sensors"], tuple(meta["windows"]),
                        meta["baseline"])
    X_last = feats.loc[dp.last_cycle_rows(feats), meta["feature_cols"]]

    pred = np.clip(model.predict(X_last), 0, meta["rul_clip"])
    truth = np.clip(rul_truth, 0, meta["rul_clip"])
    constant = np.full_like(truth, meta["train_mean_rul"], dtype=float)
    return {
        "test_engines": int(len(truth)),
        "test_rmse": rmse(truth, pred),
        "test_nasa_score": nasa_score(truth, pred),
        "baseline_constant_rmse": rmse(truth, constant),
        "baseline_constant_nasa_score": nasa_score(truth, constant),
    }, truth, pred


def load_model(model_dir=MODEL_DIR):
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_dir, "rul_xgb.json"))
    return model, meta


def main():
    model, meta = load_model()
    res, _, _ = evaluate_test(model, meta)
    print(f"[TEST]  engines={res['test_engines']}  RMSE={res['test_rmse']:.2f} cycles  "
          f"NASA score={res['test_nasa_score']:.1f}")
    print(f"[BASELINE: predict training-mean RUL]  RMSE={res['baseline_constant_rmse']:.2f}  "
          f"NASA score={res['baseline_constant_nasa_score']:.1f}")


if __name__ == "__main__":
    main()
