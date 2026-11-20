"""
Plot predicted vs true RUL at every cycle of one held-out engine, with the
WARNING / CRITICAL bands and the cycles where each alert first fired
(docs/unit_history.png).  Needs matplotlib:  python plot_history.py [unit]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import data_prep as dp  # noqa: E402
from inference import CRITICAL_RUL, WARN_RUL, alert_for, get_predictor  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "unit_history.png")


def main(unit=None):
    pred = get_predictor()
    meta = pred.meta
    unit = int(unit if unit is not None else meta["val_units"][0])
    df = dp.load_raw("train")
    eng = df[df.unit == unit].sort_values("cycle")
    life = len(eng)
    series = pred.predict_series(eng[dp.SENSOR_COLS].to_dict("records"))
    cycles = [c for c, _ in series]
    predicted = [r for _, r in series]
    truth = [min(life - c, meta["rul_clip"]) for c in cycles]

    first = {}
    for c, r in series:
        level, _ = alert_for(r)
        if level != "OK" and level not in first:
            first[level] = (c, r)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ax.axhspan(0, CRITICAL_RUL, color="#e53935", alpha=0.12, linewidth=0)
    ax.axhspan(CRITICAL_RUL, WARN_RUL, color="#ff7a00", alpha=0.10, linewidth=0)
    ax.plot(cycles, truth, color="#5f6670", linewidth=1.8, linestyle="--", label="True RUL")
    ax.plot(cycles, predicted, color="#2a6fdb", linewidth=2, label="Predicted RUL")
    colors = {"WARNING": "#ff7a00", "CRITICAL": "#e53935"}
    for level, (c, r) in first.items():
        ax.scatter([c], [r], s=60, color=colors[level], edgecolor="white", zorder=4,
                   label=f"First {level.lower()}: {life - c} cycles before failure")
    ax.set_xlabel("Operating cycle")
    ax.set_ylabel("RUL (cycles, capped at 125)")
    ax.set_title(f"Held-out FD001 engine {unit}: failed at cycle {life}", loc="left", fontsize=11)
    ax.set_xlim(cycles[0], life)
    ax.set_ylim(0, meta["rul_clip"])
    ax.grid(axis="y", color="#eceef1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {os.path.abspath(OUT)}")
    print(json.dumps({lvl: {"cycle": c, "cycles_before_failure": life - c}
                      for lvl, (c, _) in first.items()}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
