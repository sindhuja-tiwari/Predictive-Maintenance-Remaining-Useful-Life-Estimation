# Predictive Maintenance — Remaining Useful Life (RUL) for Industrial Equipment

Estimating the **Remaining Useful Life** of turbofan engines from time-series
sensor data to move maintenance from *reactive* (fix after failure) and
*scheduled* (fix on a fixed calendar, wasting healthy component life) to
**condition-based / predictive** — servicing each unit exactly when its measured
degradation warrants it. This is the maintenance strategy Siemens frames its
industrial digitalization and MindSphere/Senseye offerings around: using sensor
telemetry and **degradation modeling** to **prevent unplanned downtime**.

## Dataset
NASA **C-MAPSS Turbofan Degradation** (subset FD001). Each engine runs from
healthy operation to failure while 21 sensors + 3 operating settings are logged
per cycle. The label is RUL = cycles remaining until failure.

> A **synthetic generator** (`src/make_synthetic_data.py`) produces data in the
> exact C-MAPSS format so the pipeline runs immediately. To use the real data,
> download FD001 from the NASA Prognostics Data Repository and drop
> `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` into `data/`.

## Approach
1. **RUL labeling** — piecewise-linear target clipped at 125 cycles (an engine is
   "as-good-as-new" until wear begins; this is the standard C-MAPSS treatment).
2. **Degradation feature engineering** — per-engine rolling mean/std (windows of
   5 and 15 cycles) to expose sensor *drift* over time, plus delta-from-initial
   readings to capture cumulative wear. Windows are grouped per unit so no signal
   leaks across engines.
3. **Model** — XGBoost regressor. Strong tabular baseline, trains in seconds,
   exports to a ~200KB artifact suited to edge inference.
4. **Evaluation** — RMSE (cycles) and the **NASA asymmetric score**, which
   penalizes *late* predictions (over-estimating RUL) more than early ones,
   reflecting the real cost of an unplanned outage.

## Run
```bash
pip install -r requirements.txt
cd src
python make_synthetic_data.py   # or drop real FD001 files into ../data
python train.py                 # -> models/rul_xgb.json + meta.json
python evaluate.py              # test-set RMSE + NASA score
```

## Edge inference service (bonus)
A single-file Flask app loads the model once and serves RUL predictions with a
tiered maintenance alert (OK / WARNING / CRITICAL). Small footprint simulates a
gateway box running inference next to the machinery instead of streaming raw
sensors to the cloud.

```bash
python service/app.py           # serves on :8000
curl localhost:8000/health
# POST recent cycles -> {"predicted_rul_cycles": 42.0, "alert": "WARNING", ...}
```

Containerized for deployment:
```bash
docker build -t rul-edge .
docker run -p 8000:8000 rul-edge
```

## Project layout
```
pdm/
├── data/                     C-MAPSS files (synthetic or real)
├── src/
│   ├── make_synthetic_data.py
│   ├── data_prep.py          load, RUL labels, feature engineering
│   ├── train.py              XGBoost training + metrics
│   └── evaluate.py           official test-split evaluation
├── models/                   trained model + metadata
├── service/app.py            Flask edge-inference API
├── Dockerfile
└── requirements.txt
```

## Extending
- Add FD002–FD004 (multiple operating conditions / fault modes).
- Swap in an LSTM/1D-CNN sequence model and compare against the XGBoost baseline.
- Add a simple dashboard plotting predicted RUL and alert history per unit.
