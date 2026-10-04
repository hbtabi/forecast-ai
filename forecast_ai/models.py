"""
Three forecasters with one tiny interface:

    model.fit(history_df)            # df with columns date, value (+ optional promo, holiday)
    model.predict(future_df) -> arr  # future_df has the dates (+ known promo/holiday flags)

1. SeasonalNaive     - "same day last week". The baseline every model must beat.
2. HoltWinters       - triple exponential smoothing (level + damped trend + weekly season),
                       implemented in numpy, smoothing parameters fitted with L-BFGS-B.
3. GradientBoosting  - scikit-learn HistGradientBoosting on lag, rolling and calendar
                       features, forecasting recursively one day at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor


class Forecaster:
    name = "base"

    def fit(self, history: pd.DataFrame) -> Forecaster:
        raise NotImplementedError

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- baseline
class SeasonalNaive(Forecaster):
    """Repeat the last observed season (default: the last 7 days)."""

    name = "Seasonal naive"

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, history: pd.DataFrame) -> SeasonalNaive:
        self.last = history["value"].to_numpy()[-self.season :]
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.resize(self.last, len(future)).astype(float)


# ----------------------------------------------------------------------- Holt-Winters
def _hw_filter(y: np.ndarray, alpha: float, beta: float, gamma: float, phi: float, m: int):
    """Run additive Holt-Winters over y. Returns one-step-ahead fitted values and final state."""
    n = len(y)
    level = y[:m].mean()
    trend = (y[m : 2 * m].mean() - y[:m].mean()) / m
    season = list(y[:m] - level)
    fitted = np.empty(n)
    for t in range(n):
        s = season[t]
        fitted[t] = level + phi * trend + s
        new_level = alpha * (y[t] - s) + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        season.append(gamma * (y[t] - new_level) + (1 - gamma) * s)
        level = new_level
    return fitted, level, trend, np.array(season[-m:])


class HoltWinters(Forecaster):
    """
    Additive Holt-Winters with a damped trend. With `log=True` (default) it models
    log(value), which makes the seasonality multiplicative: weekends are "+30%",
    not "+300 units", which is what retail data actually looks like.
    """

    name = "Holt-Winters"

    def __init__(self, season: int = 7, log: bool = True, damped: bool = True):
        self.m, self.log, self.damped = season, log, damped

    def _transform(self, y):
        return np.log(np.maximum(y, 1e-6)) if self.log else y

    def fit(self, history: pd.DataFrame) -> HoltWinters:
        y = self._transform(history["value"].to_numpy(dtype=float))
        if len(y) < 3 * self.m:
            raise ValueError("Holt-Winters needs at least three seasons of history")

        def sse(params):
            a, b, g, p = params
            fitted, *_ = _hw_filter(y, a, b, g, p if self.damped else 1.0, self.m)
            err = y[self.m :] - fitted[self.m :]  # skip the warm-up season
            return float(err @ err)

        res = minimize(sse, x0=[0.3, 0.02, 0.1, 0.95], method="L-BFGS-B",
                       bounds=[(0.01, 0.99), (0.0, 0.5), (0.01, 0.99), (0.8, 0.999)])
        self.alpha, self.beta, self.gamma, self.phi = (float(v) for v in res.x)
        if not self.damped:
            self.phi = 1.0
        _, self.level, self.trend, self.season = _hw_filter(y, self.alpha, self.beta, self.gamma, self.phi, self.m)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        h = np.arange(1, len(future) + 1)
        damp = np.cumsum(self.phi ** h)                          # phi + phi^2 + ... + phi^h
        pred = self.level + damp * self.trend + self.season[(h - 1) % self.m]
        return np.exp(pred) if self.log else pred

    def params(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "phi": self.phi}


# ------------------------------------------------------------------ gradient boosting
LAGS = (1, 2, 3, 7, 14, 21, 28)


def _row_features(values: np.ndarray, t: int, d: pd.Timestamp, promo: int, holiday: int) -> list[float]:
    """
    Features for predicting day t from values[:t].

    Everything is expressed relative to the mean of the last 28 days ("base"), so the
    model learns *shape* (weekday pattern, holiday lift, promo lift) rather than absolute
    level. Trees can't extrapolate beyond the values they saw in training; ratios let a
    growing business still get sensible forecasts.
    """
    base = values[t - 28 : t].mean()
    lr = lambda v: float(np.log(max(v, 1e-6) / base))  # noqa: E731
    feats = [lr(values[t - k]) for k in LAGS]
    feats.append(lr(values[t - 7 : t].mean()))
    feats.append(lr(values[t - 364]) if t >= 364 else np.nan)   # same weekday last year
    feats.append(lr(values[t - 371 : t - 357].mean()) if t >= 371 else np.nan)  # last year's level here
    doy = d.dayofyear
    feats += [d.dayofweek, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
              d.month, d.day, promo, holiday]
    return feats


FEATURE_NAMES = [f"lag_{k}" for k in LAGS] + ["mean_7", "lag_364", "last_year_level",
                                              "day_of_week", "doy_sin", "doy_cos", "month", "day_of_month",
                                              "promo", "holiday"]


def _flags(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy(dtype=int) if col in df else np.zeros(len(df), dtype=int)


class GradientBoosting(Forecaster):
    """HistGradientBoostingRegressor on relative lag + calendar features, recursive multi-step."""

    name = "Gradient boosting"

    def __init__(self, random_state: int = 0, **params):
        defaults = {"max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                    "l2_regularization": 1.0, "categorical_features": [FEATURE_NAMES.index("day_of_week")]}
        defaults.update(params)
        self.model = HistGradientBoostingRegressor(random_state=random_state, **defaults)

    def _design(self, df: pd.DataFrame):
        values = df["value"].to_numpy(dtype=float)
        promo, hol = _flags(df, "promo"), _flags(df, "holiday")
        X, y = [], []
        for t in range(28, len(df)):
            X.append(_row_features(values, t, df["date"].iloc[t], promo[t], hol[t]))
            y.append(np.log(max(values[t], 1e-6) / values[t - 28 : t].mean()))
        return np.array(X), np.array(y)

    def fit(self, history: pd.DataFrame) -> GradientBoosting:
        if len(history) < 60:
            raise ValueError("Gradient boosting needs at least 60 days of history")
        X, y = self._design(history)
        self.model.fit(X, y)
        self.X_train, self.y_train = X, y
        self.history = history["value"].to_numpy(dtype=float)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        values = list(self.history)
        promo, hol = _flags(future, "promo"), _flags(future, "holiday")
        out = []
        for i, d in enumerate(future["date"]):
            arr = np.asarray(values)
            t = len(arr)
            x = np.array([_row_features(arr, t, pd.Timestamp(d), promo[i], hol[i])])
            ratio = float(self.model.predict(x)[0])
            pred = float(np.exp(ratio) * arr[t - 28 : t].mean())
            out.append(pred)
            values.append(pred)  # recursive: today's forecast becomes tomorrow's lag-1
        return np.array(out)


def all_models() -> list[Forecaster]:
    return [SeasonalNaive(), HoltWinters(), GradientBoosting()]
