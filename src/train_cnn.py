"""
Train a 1D-CNN on raw sensor windows and compare it with the XGBoost model.

Fair comparison: same data, same validation engines (read from models/meta.json,
so they are the ones XGBoost was early-stopped on), same RUL cap, same metrics
(RMSE and NASA score on the last cycle of the 100 official test engines).

The CNN sees the last WINDOW cycles of the 14 informative sensors as raw
readings, minus the fleet healthy baseline and divided by the training standard
deviation. It does no hand-built rolling features: the convolutions have to find
the trend themselves. Neural nets vary from seed to seed, so it trains several
and reports the mean and spread.

    pip install -r requirements-cnn.txt
    cd src && python train_cnn.py --seeds 5

Writes reports/cnn_vs_xgb.json and prints a markdown table for the README.
"""
import argparse
import json
import os

import numpy as np

import data_prep as dp
from metrics import nasa_score, rmse

WINDOW = 30
MODEL_DIR = os.environ.get(
    "RUL_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
REPORT = os.path.join(os.path.dirname(__file__), "..", "reports", "cnn_vs_xgb.json")


def normalise(df, sensors, baseline, scale):
    base = np.array([baseline[s] for s in sensors])
    return ((df[sensors].values - base) / scale).astype(np.float32)


def make_windows(df, sensors, baseline, scale, window=WINDOW, last_only=False):
    """One window per cycle: the last `window` readings up to and including that
    cycle. Cycles before `window` readings exist are padded by repeating the
    engine's first reading, so every training row can be used.
    Returns (X, rows) with X shaped (n, window, n_sensors) and `rows` the
    index into the sorted frame each window belongs to."""
    df = df.sort_values(dp.INDEX_COLS).reset_index(drop=True)
    values = normalise(df, sensors, baseline, scale)
    xs, rows = [], []
    for _, idx in df.groupby("unit").indices.items():
        v = values[idx]
        padded = np.concatenate([np.repeat(v[:1], window - 1, axis=0), v])
        ends = [len(v) - 1] if last_only else range(len(v))
        for e in ends:
            xs.append(padded[e:e + window])
            rows.append(idx[e])
    return np.stack(xs), np.array(rows), df


def build_model(n_sensors):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_sensors, 32, 5, padding=2), nn.ReLU(),
                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
                nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            )
            self.head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):                       # x: (batch, window, sensors)
            h = self.conv(x.transpose(1, 2))        # (batch, 64, window)
            h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)   # average + peak over time
            return self.head(h).squeeze(1)

    return Net()


def train_one(seed, Xtr, ytr, Xva, yva, clip, epochs=60, patience=8):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    net = build_model(Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr / clip)
    xv = torch.from_numpy(Xva)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(net(xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = np.clip(net(xv).numpy() * clip, 0, clip)
        v = rmse(yva, pv)
        if v < best - 1e-3:
            best, stale = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def predict(net, X, clip):
    import torch
    with torch.no_grad():
        return np.clip(net(torch.from_numpy(X)).numpy() * clip, 0, clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    sensors, baseline, clip = meta["sensors"], meta["baseline"], meta["rul_clip"]

    train_raw = dp.add_rul(dp.load_raw("train"))
    scale = train_raw[sensors].std().values + 1e-8
    Xall, rows, train_sorted = make_windows(train_raw, sensors, baseline, scale)
    yall = train_sorted["RUL"].values[rows].astype(np.float32)
    units = train_sorted["unit"].values[rows]
    is_val = np.isin(units, meta["val_units"])
    Xtr, ytr, Xva, yva = Xall[~is_val], yall[~is_val], Xall[is_val], yall[is_val]

    test_raw = dp.load_raw("test")
    Xte, _, _ = make_windows(test_raw, sensors, baseline, scale, last_only=True)
    truth = np.clip(dp.load_rul_truth(), 0, clip)

    runs = []
    for seed in range(args.seeds):
        net = train_one(seed, Xtr, ytr, Xva, yva, clip, epochs=args.epochs)
        pt = predict(net, Xte, clip)
        runs.append({"seed": seed,
                     "val_rmse": rmse(yva, predict(net, Xva, clip)),
                     "test_rmse": rmse(truth, pt),
                     "test_nasa_score": nasa_score(truth, pt)})
        print(f"seed {seed}: val {runs[-1]['val_rmse']:.2f}  "
              f"test RMSE {runs[-1]['test_rmse']:.2f}  NASA {runs[-1]['test_nasa_score']:.0f}")

    def stat(key):
        a = np.array([r[key] for r in runs])
        return float(a.mean()), float(a.std())

    result = {
        "window": WINDOW, "seeds": args.seeds, "runs": runs,
        "cnn": {k: stat(k) for k in ("val_rmse", "test_rmse", "test_nasa_score")},
        "xgboost": {k: meta[k] for k in ("val_rmse", "test_rmse", "test_nasa_score")},
    }
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(result, f, indent=2)

    c, x = result["cnn"], result["xgboost"]
    print("\n| Model | Val RMSE | Test RMSE | Test NASA score |")
    print("| --- | --- | --- | --- |")
    print(f"| XGBoost | {x['val_rmse']:.1f} | {x['test_rmse']:.1f} | {x['test_nasa_score']:.0f} |")
    print(f"| 1D-CNN (mean of {args.seeds} seeds) | "
          f"{c['val_rmse'][0]:.1f} ± {c['val_rmse'][1]:.1f} | "
          f"{c['test_rmse'][0]:.1f} ± {c['test_rmse'][1]:.1f} | "
          f"{c['test_nasa_score'][0]:.0f} ± {c['test_nasa_score'][1]:.0f} |")
    print(f"wrote {os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
