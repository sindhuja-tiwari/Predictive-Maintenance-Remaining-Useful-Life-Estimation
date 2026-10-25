"""Tests for the Flask edge service, using the committed model and real FD001 data."""
import numpy as np
import pytest

import data_prep as dp


@pytest.fixture(scope="module")
def client():
    import app as service_app
    return service_app.app.test_client()


@pytest.fixture(scope="module")
def test_split():
    df = dp.load_raw("test")
    return df, dp.load_rul_truth()


def _cycles(df, unit, last=None):
    eng = df[df.unit == unit]
    if last:
        eng = eng.tail(last)
    return eng[dp.SENSOR_COLS].to_dict("records")


def test_health_reports_required_sensors_and_test_metrics(client):
    d = client.get("/health").get_json()
    assert d["status"] == "up"
    assert "sensor_1" not in d["required_sensors"]
    assert d["test_rmse"] < 25


def test_short_window_gives_same_answer_as_full_history(client, test_split):
    df, _ = test_split
    for unit in (2, 20, 31):
        full = client.post("/predict", json={"cycles": _cycles(df, unit)}).get_json()
        short = client.post("/predict", json={"cycles": _cycles(df, unit, last=20)}).get_json()
        assert full["predicted_rul_cycles"] == short["predicted_rul_cycles"]


def test_engine_near_failure_is_flagged(client, test_split):
    df, truth = test_split
    unit = int(np.argmin(truth)) + 1          # engine closest to failure
    d = client.post("/predict", json={"unit_id": unit, "cycles": _cycles(df, unit, last=30)}).get_json()
    assert truth[unit - 1] <= 20
    assert d["alert"] in ("CRITICAL", "WARNING")
    assert d["predicted_rul_cycles"] < 50


def test_healthy_engine_is_ok(client, test_split):
    df, truth = test_split
    unit = int(np.argmax(truth)) + 1
    d = client.post("/predict", json={"cycles": _cycles(df, unit, last=30)}).get_json()
    assert d["alert"] == "OK"


@pytest.mark.parametrize("body", [
    None,
    [],
    {"cycles": []},
    {"cycles": "abc"},
    {"cycles": [{"sensor_2": 1.0}] * 20},
    {"cycles": [{s: 1.0 for s in dp.SENSOR_COLS}] * 3},
])
def test_bad_input_returns_400(client, body):
    r = client.post("/predict", json=body) if body is not None else \
        client.post("/predict", data="null", content_type="application/json")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_non_numeric_reading_returns_400(client, test_split):
    df, _ = test_split
    cycles = _cycles(df, 1, last=20)
    cycles[-1]["sensor_3"] = "hot"
    r = client.post("/predict", json={"cycles": cycles})
    assert r.status_code == 400
    assert "sensor_3" in r.get_json()["error"]


def test_dashboard_and_demo_sample(client):
    assert client.get("/").status_code == 200
    import app as service_app
    unit = service_app.DEMO_UNITS[0]
    s = client.get(f"/demo/sample?unit={unit}&wear=90").get_json()
    assert len(s["cycles"]) == 30
    d = client.post("/predict", json={"cycles": s["cycles"]}).get_json()
    assert abs(d["predicted_rul_cycles"] - s["true_rul_cycles"]) < 40
    assert client.get("/demo/sample?unit=99999").status_code == 400
