"""Matplotlib charts (headless, dark theme to match the README)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import Comparison  # noqa: E402

COLORS = {"Actual": "#e6edf3", "Seasonal naive": "#8b949e", "Holt-Winters": "#58a6ff", "Gradient boosting": "#f97316"}
BG = "#0d1117"


def _style(ax, title: str):
    ax.set_facecolor(BG)
    ax.set_title(title, loc="left", fontsize=12, color="#e6edf3", pad=10)
    ax.grid(alpha=0.12)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.tick_params(colors="#8b949e")


def _fig(w=10, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    fig.patch.set_facecolor(BG)
    return fig, ax


def forecast_chart(df: pd.DataFrame, comp: Comparison, path: str | Path, context_days: int = 98) -> Path:
    """Actuals for the last few months plus every model's forecast for the most recent fold."""
    fc = comp.fold_forecasts(0)
    h = comp.horizon
    tail = df.iloc[-(context_days + h):]
    test_dates = df["date"].iloc[-h:]
    fig, ax = _fig()
    ax.plot(tail["date"], tail["value"], color=COLORS["Actual"], lw=1.4, label="Actual")
    ax.axvspan(test_dates.iloc[0], test_dates.iloc[-1], color="#f97316", alpha=0.06)
    table = comp.table()
    for name, pred in fc.items():
        mape = table.loc[name, "MAPE"]
        ax.plot(test_dates, pred, color=COLORS.get(name), lw=2.2 if name != "Seasonal naive" else 1.4,
                ls="--" if name == "Seasonal naive" else "-", label=f"{name} (MAPE {mape:.1f}%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    unit = "£ / day" if comp.dataset == "sales" else "visitors / day" if comp.dataset == "footfall" else "value"
    ax.set_ylabel(unit, color="#8b949e")
    _style(ax, f"{comp.dataset}: {h}-day holdout forecast vs actuals")
    ax.legend(frameon=False, labelcolor="#e6edf3", loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return Path(path)


def metrics_chart(comp: Comparison, path: str | Path) -> Path:
    """Mean MAPE per model (bar) with each backtest fold shown as a dot."""
    rows = pd.DataFrame([{"model": f.model, "fold": f.fold, "MAPE": f.metrics["MAPE"]} for f in comp.folds])
    models = list(comp.table().index[::-1])          # best model on top
    fig, ax = _fig(10, 3.6)
    means = [rows[rows.model == m]["MAPE"].mean() for m in models]
    ax.barh(models, means, height=0.55, color=[COLORS.get(m, "#f97316") for m in models], alpha=0.85)
    for i, m in enumerate(models):
        pts = rows[rows.model == m]["MAPE"].to_numpy()
        ax.scatter(pts, np.full(len(pts), i), s=26, color="#e6edf3", zorder=3, edgecolor=BG, linewidth=0.8,
                   label="individual folds" if i == 0 else None)
        ax.text(max(pts.max(), means[i]) + 0.3, i, f"{means[i]:.1f}%", va="center", color="#e6edf3", fontsize=10)
    ax.tick_params(axis="y", colors="#e6edf3")
    ax.set_xlabel("MAPE (%), lower is better", color="#8b949e")
    ax.set_xlim(0, rows["MAPE"].max() * 1.18)
    n_folds = rows["fold"].nunique()
    _style(ax, f"{comp.dataset}: rolling-origin backtest, mean of {n_folds} folds × {comp.horizon} days")
    ax.legend(frameon=False, labelcolor="#8b949e", loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return Path(path)


def future_chart(df: pd.DataFrame, future: pd.DataFrame, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 model_name: str, dataset: str, path: str | Path, context_days: int = 84) -> Path:
    tail = df.iloc[-context_days:]
    fig, ax = _fig(10, 4.2)
    ax.plot(tail["date"], tail["value"], color=COLORS["Actual"], lw=1.4, label="History")
    color = COLORS.get(model_name, "#f97316")
    ax.fill_between(future["date"], lo, hi, color=color, alpha=0.18, label="80% interval")
    ax.plot(future["date"], pred, color=color, lw=2.2, label=f"Forecast ({model_name})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    _style(ax, f"{dataset}: next {len(future)} days")
    ax.legend(frameon=False, labelcolor="#e6edf3", loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return Path(path)


def importance_chart(names: list[str], values: np.ndarray, dataset: str, path: str | Path) -> Path:
    order = np.argsort(values)
    fig, ax = _fig(8, 4.4)
    ax.barh(np.array(names)[order], values[order], color="#f97316", alpha=0.85)
    ax.tick_params(axis="y", colors="#e6edf3")
    ax.set_xlabel("permutation importance (drop in R² when shuffled)", color="#8b949e")
    _style(ax, f"{dataset}: what the gradient-boosting model relies on")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return Path(path)
