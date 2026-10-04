"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p) -> float:
    """Mean absolute percentage error, in percent."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def bias(y, p) -> float:
    """Mean forecast error as a % of actuals; positive means over-forecasting."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((p.sum() - y.sum()) / y.sum() * 100)


def mase(y, p, history, season: int = 7) -> float:
    """
    Mean absolute scaled error: MAE divided by the in-sample MAE of a seasonal-naive forecast.
    < 1 means better than "same day last week".
    """
    h = np.asarray(history, float)
    scale = np.mean(np.abs(h[season:] - h[:-season]))
    return mae(y, p) / scale


def all_metrics(y, p, history) -> dict[str, float]:
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p), "MASE": mase(y, p, history), "Bias": bias(y, p)}
