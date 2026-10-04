"""
Command-line interface.

    python -m forecast_ai compare  --dataset sales --horizon 28 --folds 3
    python -m forecast_ai forecast --dataset footfall --horizon 14 --model auto
    python -m forecast_ai generate-data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR, DATASETS, load_dataset, synthetic_series, uk_bank_holidays
from .evaluate import backtest
from .models import FEATURE_NAMES, GradientBoosting, HoltWinters, SeasonalNaive
from .narrate import llm_summary
from .report import build_report, write_report
from .tables import text_table

MODELS = {"naive": SeasonalNaive, "hw": HoltWinters, "gbm": GradientBoosting}
NAME_TO_KEY = {"Seasonal naive": "naive", "Holt-Winters": "hw", "Gradient boosting": "gbm"}


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() and not os.environ.get("NO_COLOR") else s


def future_frame(df: pd.DataFrame, horizon: int, promo_dates: list[str] | None = None) -> pd.DataFrame:
    """Calendar for the next `horizon` days with known holiday flags and planned promotions."""
    dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    hols = uk_bank_holidays(range(dates[0].year, dates[-1].year + 1))
    promos = {pd.Timestamp(d) for d in (promo_dates or [])}
    return pd.DataFrame({"date": dates, "promo": [int(d in promos) for d in dates],
                         "holiday": [int(d.date() in hols) for d in dates]})


def cmd_compare(a) -> int:
    df = load_dataset(a.dataset)
    name = a.dataset if a.dataset in DATASETS else Path(a.dataset).stem
    print(_c("1", f"forecast-ai · {name} · {len(df):,} days · horizon {a.horizon} · {a.folds} folds"))
    comp = backtest(df, a.horizon, a.folds, dataset=name, log=print)
    table = comp.table()
    print()
    print(text_table(table))
    print(_c("32", f"\nbest model: {comp.best_model()}"))

    charts: dict[str, str] = {}
    hw = HoltWinters().fit(df.iloc[: -a.horizon])
    if not a.no_charts:
        from . import charts as ch

        out = Path(a.charts_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_dir = Path(a.report).parent if a.report else out
        rel = lambda p: os.path.relpath(p, report_dir).replace(os.sep, "/")  # noqa: E731
        charts["Holdout forecast"] = rel(ch.forecast_chart(df, comp, out / f"{name}_forecast.png"))
        charts["Backtest accuracy"] = rel(ch.metrics_chart(comp, out / f"{name}_backtest.png"))
        if a.importance:
            from sklearn.inspection import permutation_importance

            gbm = GradientBoosting().fit(df.iloc[: -a.horizon])
            X, y = gbm.X_train[-500:], gbm.y_train[-500:]
            imp = permutation_importance(gbm.model, X, y, n_repeats=5, random_state=0).importances_mean
            charts["Feature importance"] = rel(ch.importance_chart(FEATURE_NAMES, imp, name, out / f"{name}_importance.png"))
        print(f"charts  -> {out}/")

    source, summary = llm_summary(table, name, a.horizon)
    if a.report:
        path = write_report(build_report(df, comp, summary, source, charts, hw.params()), a.report)
        print(f"report  -> {path}")
    print(_c("36", f"\nsummary ({source}):"), summary)
    return 0


def cmd_forecast(a) -> int:
    df = load_dataset(a.dataset)
    name = a.dataset if a.dataset in DATASETS else Path(a.dataset).stem
    key = a.model
    if key == "auto":
        comp = backtest(df, a.horizon, 2, dataset=name)
        key = NAME_TO_KEY[comp.best_model()]
        print(f"auto-selected model: {comp.best_model()} (lowest MASE over 2 backtest folds)")
    else:
        comp = backtest(df, a.horizon, 2, models=[MODELS[key]()], dataset=name)

    # 80% interval from backtest errors: quantiles of log(actual / forecast) by horizon step.
    model_name = MODELS[key].name
    ratios = []
    for f in comp.folds:
        if f.model == model_name:
            start = df.index[df["date"] == f.cutoff][0]
            actual = df["value"].to_numpy()[start : start + a.horizon]
            ratios.append(np.log(actual / f.forecast))
    r = np.vstack(ratios)
    lo_q, hi_q = np.quantile(r, 0.1), np.quantile(r, 0.9)

    future = future_frame(df, a.horizon, a.promo)
    pred = MODELS[key]().fit(df).predict(future)
    widen = np.sqrt(np.arange(1, a.horizon + 1) / a.horizon) * 0.5 + 0.5   # uncertainty grows with lead time
    lo, hi = pred * np.exp(lo_q * widen), pred * np.exp(hi_q * widen)

    out = future.assign(forecast=pred.round(1), lower_80=lo.round(1), upper_80=hi.round(1))
    print(out.to_string(index=False, formatters={"date": lambda d: f"{d:%a %d %b %Y}"}))
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"\nsaved forecast -> {a.out}")
    if a.chart:
        from .charts import future_chart

        future_chart(df, future, pred, lo, hi, model_name, name, a.chart)
        print(f"saved chart    -> {a.chart}")
    return 0


def cmd_generate(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in DATASETS:
        df = synthetic_series(kind, start=a.start, days=a.days, seed=a.seed)
        df.to_csv(out / f"{kind}.csv", index=False, date_format="%Y-%m-%d")
        print(f"wrote {out / f'{kind}.csv'} ({len(df)} rows)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="forecast_ai", description="Retail time-series forecasting model comparison.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="backtest all models and write charts + a Markdown report")
    c.add_argument("--dataset", default="sales", help="sales | footfall | path/to/file.csv (date,value)")
    c.add_argument("--horizon", type=int, default=28)
    c.add_argument("--folds", type=int, default=3)
    c.add_argument("--charts-dir", default="docs")
    c.add_argument("--report", default=None, help="write a Markdown report here, e.g. reports/sales.md")
    c.add_argument("--importance", action="store_true", help="also chart gradient-boosting feature importance")
    c.add_argument("--no-charts", action="store_true")
    c.set_defaults(func=cmd_compare)

    f = sub.add_parser("forecast", help="forecast the future with one model (or the backtest winner)")
    f.add_argument("--dataset", default="sales")
    f.add_argument("--horizon", type=int, default=14)
    f.add_argument("--model", choices=["auto", *MODELS], default="auto")
    f.add_argument("--promo", nargs="*", default=[], help="planned promotion dates, YYYY-MM-DD")
    f.add_argument("--out", default=None, help="CSV output path")
    f.add_argument("--chart", default=None, help="PNG output path")
    f.set_defaults(func=cmd_forecast)

    g = sub.add_parser("generate-data", help="regenerate the bundled synthetic CSVs")
    g.add_argument("--out", default=str(DATA_DIR))
    g.add_argument("--start", default="2023-10-01")
    g.add_argument("--days", type=int, default=3 * 365)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
