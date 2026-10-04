"""Markdown report generation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .evaluate import Comparison
from .tables import markdown_table


def build_report(df: pd.DataFrame, comp: Comparison, summary: str, summary_source: str,
                 charts: dict[str, str], hw_params: dict | None = None) -> str:
    t = comp.table()
    lines = [
        f"# Forecast report: {comp.dataset}",
        "",
        f"_Generated {datetime.now():%d %b %Y %H:%M} by forecast-ai · {len(df):,} days of data "
        f"({df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}) · horizon {comp.horizon} days · "
        f"{t.shape[0]} models × {len({f.fold for f in comp.folds})} backtest folds_",
        "",
        "## Summary",
        "",
        summary,
        "",
        f"<sub>Summary written by: `{summary_source}`</sub>",
        "",
        "## Backtest results (mean across folds)",
        "",
        markdown_table(t, bold_first=True),
        "",
        "MAPE = mean absolute % error · MASE < 1 beats \"same day last week\" · Bias > 0 = over-forecasting",
        "",
    ]
    for title, path in charts.items():
        lines += [f"## {title}", "", f"![{title}]({path})", ""]
    if hw_params:
        p = ", ".join(f"{k} = {v:.3f}" for k, v in hw_params.items())
        lines += ["## Model notes", "", f"- Holt-Winters fitted smoothing parameters: {p}",
                  "- Gradient boosting uses lags 1-28, a 7-day mean, same-day-last-year and calendar/promo/holiday "
                  "features, all relative to the trailing 28-day mean so it can follow growth.", ""]
    return "\n".join(lines)


def write_report(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
