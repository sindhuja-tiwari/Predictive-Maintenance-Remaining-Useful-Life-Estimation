"""End-to-end training on a small synthetic fleet, in a temp dir."""
import json

import make_synthetic_data
import train as train_mod


def test_training_pipeline_learns_and_is_reproducible(tmp_path):
    data = tmp_path / "data"
    make_synthetic_data.make(str(data), n_train=30, n_test=20, seed=0)
    _, meta1 = train_mod.train(str(data), str(tmp_path / "m1"), n_estimators=300, verbose=False)
    _, meta2 = train_mod.train(str(data), str(tmp_path / "m2"), n_estimators=300, verbose=False)

    assert meta1["test_rmse"] == meta2["test_rmse"]
    assert (tmp_path / "m1" / "rul_xgb.json").read_bytes() == (tmp_path / "m2" / "rul_xgb.json").read_bytes()
    # the model must beat predicting the training-mean RUL for every engine
    assert meta1["test_rmse"] < meta1["baseline_constant_rmse"]
    # flat sensors are dropped, and the baseline covers every sensor used
    assert "sensor_1" not in meta1["sensors"]
    assert set(meta1["baseline"]) == set(meta1["sensors"])
    json.dumps(meta1)
