"""
Generate synthetic data in the exact NASA C-MAPSS FD001 format.

The repo ships the real FD001 files in data/, so you do not need this to run the
project. It is here for quick experiments and for the test suite, and it writes
to data/synthetic/ by default so it never overwrites the real data. Point the
pipeline at it with:  RUL_DATA_DIR=data/synthetic python train.py

Like the real fleet, all engines share one healthy baseline per sensor (with a
small per-engine manufacturing offset), and each informative sensor drifts in
a fixed direction as wear accelerates towards failure. Seven sensors are
constant, as in FD001.

C-MAPSS column layout (space-separated, no header):
  unit  cycle  op_setting_1  op_setting_2  op_setting_3  sensor_1 ... sensor_21
"""
import argparse
import os

import numpy as np

N_SENSORS = 21
CONSTANT_SENSORS = {1, 5, 6, 10, 16, 18, 19}  # flat in FD001
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")


def _fleet(rng):
    base = rng.uniform(10, 2000, N_SENSORS)
    noise_sd = base * rng.uniform(0.0005, 0.002, N_SENSORS)
    drift = base * rng.uniform(0.002, 0.01, N_SENSORS) * rng.choice([-1, 1], N_SENSORS)
    for s in CONSTANT_SENSORS:
        noise_sd[s - 1] = 0.0
        drift[s - 1] = 0.0
    return base, noise_sd, drift


def _run_to_failure(rng, fleet, unit_id, life):
    """One engine degrading over `life` cycles; wear grows exponentially."""
    base, noise_sd, drift = fleet
    cycles = np.arange(1, life + 1)
    wear = (np.exp(3 * cycles / life) - 1) / (np.exp(3) - 1)   # 0 -> 1
    offset = rng.normal(0, noise_sd)                            # per-engine offset
    sensors = (base + offset + np.outer(wear, drift)
               + rng.normal(0, 1, (life, N_SENSORS)) * noise_sd)
    ops = np.column_stack([rng.normal(0, 0.002, life), rng.normal(0, 0.0003, life),
                           np.full(life, 100.0)])
    return np.column_stack([np.full(life, unit_id), cycles, ops, sensors])


def generate(rng, fleet, n_units, min_life=130, max_life=350):
    return [_run_to_failure(rng, fleet, u, int(rng.integers(min_life, max_life)))
            for u in range(1, n_units + 1)]


def write(path, arr):
    fmt = ["%d", "%d"] + ["%.4f"] * (arr.shape[1] - 2)
    np.savetxt(path, arr, fmt=fmt)


def make(out_dir=DEFAULT_OUT, n_train=100, n_test=100, seed=42):
    rng = np.random.default_rng(seed)
    fleet = _fleet(rng)
    os.makedirs(out_dir, exist_ok=True)
    train = np.vstack(generate(rng, fleet, n_train))
    write(os.path.join(out_dir, "train_FD001.txt"), train)

    # Test set: cut each engine at a random point before failure; the true RUL
    # goes in RUL_FD001.txt, one line per unit (C-MAPSS convention).
    test_rows, rul_truth = [], []
    for eng in generate(rng, fleet, n_test):
        life = len(eng)
        cut = int(rng.integers(int(life * 0.4), int(life * 0.95)))
        test_rows.append(eng[:cut])
        rul_truth.append(life - cut)
    test = np.vstack(test_rows)
    write(os.path.join(out_dir, "test_FD001.txt"), test)
    np.savetxt(os.path.join(out_dir, "RUL_FD001.txt"), np.array(rul_truth, dtype=int), fmt="%d")
    return train.shape, test.shape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--train-units", type=int, default=100)
    ap.add_argument("--test-units", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    tr, te = make(a.out, a.train_units, a.test_units, a.seed)
    print(f"Wrote synthetic FD001 to {os.path.abspath(a.out)}: train={tr}, test={te}")
