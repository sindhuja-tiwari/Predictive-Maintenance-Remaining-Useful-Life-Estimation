"""
Generate synthetic data in the exact NASA C-MAPSS FD001 format so the pipeline
runs end-to-end without the real download. Replace data/train_FD001.txt and
data/test_FD001.txt with the real files (same columns) and everything else works
unchanged.

C-MAPSS column layout (space-separated, no header):
  unit  cycle  op_setting_1  op_setting_2  op_setting_3  sensor_1 ... sensor_21
"""
import numpy as np
import os

RNG = np.random.default_rng(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
N_SENSORS = 21


def _run_to_failure(unit_id, life):
    """One engine degrading over `life` cycles. Sensors drift as health decays."""
    cycles = np.arange(1, life + 1)
    health = 1 - (cycles / life)  # 1.0 healthy -> 0.0 failed
    op1 = RNG.normal(0, 0.002, life)
    op2 = RNG.normal(0, 0.0003, life)
    op3 = np.full(life, 100.0)

    rows = []
    # Each sensor: a baseline + a monotonic degradation signature + noise.
    base = RNG.normal(500, 150, N_SENSORS)
    drift = RNG.normal(0, 1, N_SENSORS) * RNG.choice([0, 1], N_SENSORS, p=[0.3, 0.7])
    noise_sd = np.abs(RNG.normal(0.5, 0.3, N_SENSORS))
    for i, c in enumerate(cycles):
        sensors = base + drift * (1 - health[i]) * 30 + RNG.normal(0, noise_sd)
        rows.append([unit_id, c, op1[i], op2[i], op3[i], *sensors])
    return rows


def generate(n_units, seed_offset=0, min_life=130, max_life=350):
    rows = []
    for u in range(1, n_units + 1):
        life = int(RNG.integers(min_life, max_life))
        rows.extend(_run_to_failure(u + seed_offset, life))
    return np.array(rows, dtype=float)


def write(path, arr):
    fmt = ["%d", "%d"] + ["%.4f"] * (arr.shape[1] - 2)
    np.savetxt(path, arr, fmt=fmt)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    train = generate(100)
    write(os.path.join(DATA_DIR, "train_FD001.txt"), train)

    # Test set: truncate each engine at a random point before failure; RUL is the
    # remaining cycles, stored in a separate RUL_FD001.txt file (C-MAPSS convention).
    full = generate(100, seed_offset=1000)
    test_rows, rul_truth = [], []
    for u in np.unique(full[:, 0]):
        eng = full[full[:, 0] == u]
        life = len(eng)
        cut = int(RNG.integers(int(life * 0.4), int(life * 0.95)))
        test_rows.append(eng[:cut])
        rul_truth.append(life - cut)
    test = np.vstack(test_rows)
    # renumber test units 1..100
    remap = {old: new for new, old in enumerate(np.unique(test[:, 0]), start=1)}
    test[:, 0] = [remap[u] for u in test[:, 0]]
    write(os.path.join(DATA_DIR, "test_FD001.txt"), test)
    np.savetxt(os.path.join(DATA_DIR, "RUL_FD001.txt"),
               np.array(rul_truth, dtype=int), fmt="%d")
    print(f"Wrote synthetic FD001: train={train.shape}, test={test.shape}")
