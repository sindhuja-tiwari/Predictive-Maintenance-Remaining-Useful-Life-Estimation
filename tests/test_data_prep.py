import numpy as np
import pandas as pd
import pytest

import data_prep as dp


def _toy(units=((1, 6), (2, 4))):
    rows = []
    for u, n in units:
        for c in range(1, n + 1):
            rows.append({"unit": u, "cycle": c, **{s: float(u * 100 + c) for s in dp.SENSOR_COLS}})
    return pd.DataFrame(rows)


def test_train_rul_counts_down_to_zero_and_is_clipped():
    df = _toy(((1, 200),))
    out = dp.add_rul(df)
    assert out["RUL"].iloc[-1] == 0
    assert out["RUL"].max() == dp.RUL_CLIP
    assert out["RUL"].iloc[-10] == 9


def test_test_rul_adds_final_truth_per_unit():
    df = _toy(((1, 5), (2, 3)))
    out = dp.add_rul(df, rul_truth=[10, 200])
    u1 = out[out.unit == 1]["RUL"].tolist()
    assert u1 == [14, 13, 12, 11, 10]
    assert out[out.unit == 2]["RUL"].tolist() == [dp.RUL_CLIP] * 3


def test_select_useful_sensors_drops_flat_and_two_valued_sensors():
    df = _toy()
    df["sensor_1"] = 518.67
    df["sensor_6"] = np.where(df["cycle"] % 2, 21.60, 21.61)
    useful = dp.select_useful_sensors(df)
    assert "sensor_1" not in useful and "sensor_6" not in useful
    assert "sensor_2" in useful


def test_rolling_features_do_not_leak_across_units():
    df = _toy()
    sensors = ["sensor_2"]
    base = {"sensor_2": 0.0}
    feats = dp.engineer(df, sensors, (5,), base)
    first_u2 = feats[(feats.unit == 2) & (feats.cycle == 1)].iloc[0]
    # first row of unit 2 only sees itself, not unit 1's last readings
    assert first_u2["sensor_2_mean5"] == first_u2["sensor_2"]
    assert first_u2["sensor_2_std5"] == 0


def test_delta_uses_fixed_baseline_not_window_start():
    """The bug this guards against: delta was computed from the first row of
    whatever window was sent, so a short late-life window looked healthy."""
    df = _toy(((1, 30),))
    sensors = ["sensor_2"]
    base = {"sensor_2": 100.0}
    full = dp.engineer(df, sensors, (5, 15), base)
    tail = dp.engineer(df.tail(15).copy(), sensors, (5, 15), base)
    cols = dp.feature_columns(sensors, (5, 15))
    pd.testing.assert_series_equal(full[cols].iloc[-1], tail[cols].iloc[-1], check_names=False)


def test_engineer_requires_baseline():
    with pytest.raises(ValueError):
        dp.engineer(_toy(), ["sensor_2"])


def test_feature_columns_match_engineer_output():
    sensors = ["sensor_2", "sensor_3"]
    feats = dp.engineer(_toy(), sensors, (5, 15), {"sensor_2": 0.0, "sensor_3": 0.0})
    for c in dp.feature_columns(sensors, (5, 15)):
        assert c in feats.columns
