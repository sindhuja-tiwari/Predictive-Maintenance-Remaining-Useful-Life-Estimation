"""
Train and test the XGBoost RUL model on C-MAPSS FD001 to FD004.

FD002 and FD004 mix six flight conditions. Those shift every sensor far more than
engine wear does, so the sensors are normalised per condition first
(data_prep.ConditionScaler). For these two subsets the script also trains the same
model WITHOUT per-condition normalisation, to show what the step is worth.

Everything else matches train.py: same 14 sensors, same rolling features, same
XGBoost settings, validation engines held out by unit (20%, seed 42) for early
stopping, and the test score taken on the last cycle of every official test engine
with true RUL capped at 125.

Put train/test/RUL files for the subsets you want in data/ (or --data-dir), e.g.
download CMAPSSData.zip from the NASA Prognostics Data Repository, then:

    cd src && python train_multi.py                        # all four subsets
    python train_multi.py --subsets FD002 FD004 --data-dir ../data

Writes reports/multi_subset.json and prints a markdown table for the README. It does
not touch models/ (the deployed service stays on FD001).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import data_prep as dp
from metrics import nasa_score, rmse
from train import SEED, WINDOWS, make_regressor

# The 14 sensors that carry signal in C-MAPSS (the set select_useful_sensors finds
# for FD001). Fixed here because with six conditions every sensor varies, so a
# "constant sensor" test can no longer pick them out.
SENSORS = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
REPORT = os.path.join(os.path.dirname(__file__), "..", "reports", "multi_subset.json")


def run(subset, n_conditions, data_dir=None):
    train_raw = dp.load_raw("train", subset, data_dir)
    test_raw = dp.load_raw("test", subset, data_dir)
    truth = np.clip(dp.load_rul_truth(subset, data_dir), 0, dp.RUL_CLIP)

    scaler = dp.ConditionScaler(n_conditions, SEED).fit(train_raw, SENSORS)
    train_n, test_n = scaler.transform(train_raw), scaler.transform(test_raw)
    baseline = dp.healthy_baseline(train_n, SENSORS)

    train_n = dp.add_rul(train_n).sort_values(dp.INDEX_COLS).reset_index(drop=True)
    feats = dp.engineer(train_n, SENSORS, WINDOWS, baseline)
    cols = dp.feature_columns(SENSORS, WINDOWS)
    df = pd.concat([feats[cols], train_n[["unit", "cycle", "RUL"]]], axis=1)

    tr_idx, va_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=df["unit"]))
    tr, va = df.iloc[tr_idx], df.iloc[va_idx]
    model = make_regressor()
    model.fit(tr[cols], tr["RUL"], eval_set=[(va[cols], va["RUL"])], verbose=False)

    val_pred = np.clip(model.predict(va[cols]), 0, dp.RUL_CLIP)
    test_feats = dp.engineer(test_n, SENSORS, WINDOWS, baseline)
    x_last = test_feats.loc[dp.last_cycle_rows(test_feats), cols]
    pred = np.clip(model.predict(x_last), 0, dp.RUL_CLIP)
    const = np.full_like(truth, tr["RUL"].mean(), dtype=float)
    return {
        "subset": subset, "n_conditions": n_conditions,
        "train_engines": int(train_raw["unit"].nunique()),
        "test_engines": int(len(truth)),
        "val_rmse": rmse(va["RUL"], val_pred),
        "test_rmse": rmse(truth, pred),
        "test_nasa_score": nasa_score(truth, pred),
        "baseline_rmse": rmse(truth, const),
        "baseline_nasa_score": nasa_score(truth, const),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    results = []
    for subset in args.subsets:
        k = dp.N_CONDITIONS[subset]
        results.append(dict(run(subset, k, args.data_dir), label=f"{subset}"))
        print(f"{subset}: test RMSE {results[-1]['test_rmse']:.2f}  "
              f"NASA {results[-1]['test_nasa_score']:.0f}", flush=True)
        if k > 1:
            ab = run(subset, 1, args.data_dir)
            results.append(dict(ab, label=f"{subset} without condition normalisation"))
            print(f"{subset} (no condition normalisation): test RMSE {ab['test_rmse']:.2f}  "
                  f"NASA {ab['test_nasa_score']:.0f}", flush=True)

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(results, f, indent=2)

    print("\n| Subset | Conditions | Train / test engines | Val RMSE | Test RMSE | "
          "Test NASA score | Predict-the-mean RMSE |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(f"| {r['label']} | {r['n_conditions']} | {r['train_engines']} / {r['test_engines']} | "
              f"{r['val_rmse']:.1f} | {r['test_rmse']:.1f} | {r['test_nasa_score']:,.0f} | "
              f"{r['baseline_rmse']:.1f} |")
    print(f"wrote {os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
