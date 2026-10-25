"""
Plot predicted vs true RUL for the 100 FD001 test engines (docs/test_predictions.png).
Needs matplotlib:  pip install matplotlib && python plot_results.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluate import evaluate_test, load_model  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "test_predictions.png")


def main():
    model, meta = load_model()
    res, truth, pred = evaluate_test(model, meta)
    order = np.argsort(truth)
    x = np.arange(len(truth))

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ax.vlines(x, truth[order], pred[order], color="#c9ccd1", linewidth=1, zorder=1)
    ax.plot(x, truth[order], color="#5f6670", linewidth=2, label="True RUL", zorder=2)
    ax.scatter(x, pred[order], s=16, color="#2a6fdb", edgecolor="white", linewidth=0.6,
               label="Predicted RUL", zorder=3)
    ax.set_xlabel("Test engines, sorted by true RUL")
    ax.set_ylabel("RUL (cycles, capped at 125)")
    ax.set_title(f"FD001 test split: RMSE {res['test_rmse']:.1f} cycles, "
                 f"NASA score {res['test_nasa_score']:.0f}", loc="left", fontsize=11)
    ax.grid(axis="y", color="#eceef1", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
