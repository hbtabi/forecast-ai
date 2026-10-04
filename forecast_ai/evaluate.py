"""Rolling-origin backtesting: train on the past, forecast the next `horizon` days, step back, repeat."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import all_metrics
from .models import Forecaster, all_models


@dataclass
class FoldResult:
    model: str
    fold: int
    cutoff: pd.Timestamp
    metrics: dict[str, float]
    forecast: np.ndarray
    seconds: float


@dataclass
class Comparison:
    dataset: str
    horizon: int
    folds: list[FoldResult] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        """Mean metrics per model across folds, best (lowest MASE) first."""
        rows = [{"model": f.model, "fold": f.fold, **f.metrics, "fit_s": f.seconds} for f in self.folds]
        df = pd.DataFrame(rows).drop(columns="fold").groupby("model", sort=False).mean()
        return df.sort_values("MASE")

    def best_model(self) -> str:
        return str(self.table().index[0])

    def fold_forecasts(self, fold: int = 0) -> dict[str, np.ndarray]:
        return {f.model: f.forecast for f in self.folds if f.fold == fold}


def cutoffs(n: int, horizon: int, folds: int) -> list[int]:
    """Index where each fold's test window starts; fold 0 is the most recent."""
    return [n - horizon * (k + 1) for k in range(folds)]


def backtest(df: pd.DataFrame, horizon: int = 28, folds: int = 3, models: list[Forecaster] | None = None,
             dataset: str = "custom", log=lambda s: None) -> Comparison:
    if len(df) < horizon * folds + 400:
        folds = max(1, (len(df) - 400) // horizon)
    comp = Comparison(dataset=dataset, horizon=horizon)
    for k, start in enumerate(cutoffs(len(df), horizon, folds)):
        train, test = df.iloc[:start], df.iloc[start : start + horizon]
        future = test.drop(columns="value")
        for model in models or all_models():
            t0 = time.perf_counter()
            pred = model.fit(train).predict(future)
            secs = time.perf_counter() - t0
            m = all_metrics(test["value"].to_numpy(), pred, train["value"].to_numpy())
            comp.folds.append(FoldResult(model.name, k, test["date"].iloc[0], m, pred, secs))
            log(f"  fold {k} · {model.name:<18} MAPE {m['MAPE']:5.2f}%  MASE {m['MASE']:.3f}  ({secs:.1f}s)")
    return comp
