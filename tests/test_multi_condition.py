"""ConditionScaler and the CNN windowing (numpy only, no torch needed)."""
import json

import numpy as np
import pandas as pd

import data_prep as dp


def _fleet(n_per=200, seed=0):
    """Two flight conditions with very different sensor levels."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in (1, 2):
        for cycle in range(1, n_per + 1):
            high = cycle % 2 == 0
            op = (35.0, 0.84, 100.0) if high else (0.0, 0.0, 100.0)
            base = 500.0 if high else 100.0
            rows.append([unit, cycle, *op, base + rng.normal(0, 2),
                         base * 2 + rng.normal(0, 4)])
    return pd.DataFrame(rows, columns=["unit", "cycle", *dp.OP_COLS, "sensor_2", "sensor_3"])


def test_condition_scaler_removes_the_condition_offset():
    df = _fleet()
    sc = dp.ConditionScaler(2).fit(df, ["sensor_2", "sensor_3"])
    cond = sc.condition(df)
    assert sorted(np.bincount(cond)) == [200, 200]          # both conditions found
    z = sc.transform(df)
    for k in (0, 1):
        part = z[cond == k]
        assert abs(part["sensor_2"].mean()) < 1e-6
        assert abs(part["sensor_2"].std() - 1) < 0.05
    assert df["sensor_2"].std() > 100                       # the raw spread was huge


def test_condition_scaler_does_not_change_the_input_and_serialises():
    df = _fleet()
    before = df.copy()
    sc = dp.ConditionScaler(2).fit(df, ["sensor_2", "sensor_3"])
    sc.transform(df)
    pd.testing.assert_frame_equal(df, before)
    json.dumps(sc.to_dict())                                # plain lists, safe for meta.json


def test_single_condition_is_plain_standardisation():
    df = _fleet()
    z = dp.ConditionScaler(1).fit(df, ["sensor_2"]).transform(df)
    assert abs(z["sensor_2"].mean()) < 1e-6
    assert abs(z["sensor_2"].std() - 1) < 0.01


def test_cnn_windows_are_padded_and_end_on_the_right_cycle():
    from train_cnn import make_windows
    df = dp.add_rul(dp.load_raw("train"))
    sensors = dp.select_useful_sensors(df)
    baseline = dp.healthy_baseline(df, sensors)
    scale = df[sensors].std().values + 1e-8
    X, rows, ordered = make_windows(df, sensors, baseline, scale, window=30)
    assert X.shape == (len(ordered), 30, len(sensors))
    first = np.where(ordered["unit"].values[rows] == 1)[0][0]
    assert np.allclose(X[first][0], X[first][-1])           # padded by repeating cycle 1
    Xl, rl, _ = make_windows(df, sensors, baseline, scale, window=30, last_only=True)
    assert len(Xl) == df["unit"].nunique()
