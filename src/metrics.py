"""Evaluation metrics for RUL prediction."""
import numpy as np


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred):
    """PHM08 / C-MAPSS asymmetric score (lower is better), summed over engines.

    d = predicted - true. Early predictions (d < 0) cost exp(-d/13) - 1 and late
    predictions (d > 0) cost exp(d/10) - 1, so being late is punished harder:
    20 cycles late costs about 6.4, 20 cycles early about 3.7.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
