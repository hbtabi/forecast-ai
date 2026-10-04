"""
Datasets: a realistic synthetic generator for daily retail sales and store footfall,
plus a loader for your own CSV files (columns: date, value).

The synthetic series combine the patterns real retail data has, so the models have
something meaningful to learn:
  * slow trend growth
  * weekly seasonality (busy weekends for footfall, Friday/Saturday peaks for sales)
  * yearly seasonality (summer lull, December peak)
  * UK bank holidays and occasional promotions
  * multiplicative noise and the odd weather shock
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS = ("sales", "footfall")


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher) for Easter Sunday."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _first_weekday(year: int, month: int, weekday: int, last: bool = False) -> date:
    if last:
        d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        return d
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


def uk_bank_holidays(years) -> set[date]:
    """England & Wales bank holidays (without one-off royal events)."""
    out: set[date] = set()
    for y in years:
        easter = _easter(y)
        out |= {
            date(y, 1, 1), easter - timedelta(days=2), easter + timedelta(days=1),
            _first_weekday(y, 5, 0), _first_weekday(y, 5, 0, last=True), _first_weekday(y, 8, 0, last=True),
            date(y, 12, 25), date(y, 12, 26),
        }
    return out


def synthetic_series(kind: str = "sales", start: str = "2023-10-01", days: int = 3 * 365, seed: int = 42) -> pd.DataFrame:
    """Generate a daily series with columns: date, value, promo, holiday."""
    if kind not in DATASETS:
        raise ValueError(f"unknown dataset {kind!r}; choose from {DATASETS}")
    rng = np.random.default_rng(seed + (0 if kind == "sales" else 1))
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()
    holidays = uk_bank_holidays(range(dates[0].year, dates[-1].year + 1))
    is_hol = np.array([d.date() in holidays for d in dates])

    if kind == "sales":
        base, growth = 1200.0, 0.00035                         # £ per day, ~13% growth a year
        weekly = np.array([0.86, 0.88, 0.93, 0.98, 1.12, 1.28, 0.95])[dow]
        yearly = 1 + 0.10 * np.cos(2 * np.pi * (doy - 350) / 365.25)  # December peak
        holiday_effect = np.where(is_hol, 1.25, 1.0)
    else:
        base, growth = 2600.0, 0.00015                         # visitors per day
        weekly = np.array([0.78, 0.80, 0.84, 0.90, 1.05, 1.42, 1.21])[dow]
        yearly = 1 + 0.14 * np.sin(2 * np.pi * (doy - 80) / 365.25)   # summer high street peak
        holiday_effect = np.where(is_hol, 1.35, 1.0)

    # Promotions: ~2 short campaigns a month.
    promo = np.zeros(days, dtype=bool)
    for s in rng.choice(days - 4, size=days // 15, replace=False):
        promo[s : s + rng.integers(2, 5)] = True
    promo_effect = np.where(promo, 1.18 if kind == "sales" else 1.08, 1.0)

    # Weather shocks: occasional bad-weather days cut demand.
    shock = np.where(rng.random(days) < 0.03, rng.uniform(0.6, 0.85, days), 1.0)
    noise = rng.lognormal(0, 0.06, days)

    value = base * np.exp(growth * t) * weekly * yearly * holiday_effect * promo_effect * shock * noise
    return pd.DataFrame({"date": dates, "value": value.round(0 if kind == "footfall" else 2),
                         "promo": promo.astype(int), "holiday": is_hol.astype(int)})


def load_dataset(name_or_path: str) -> pd.DataFrame:
    """Load a bundled dataset by name ("sales"/"footfall") or any CSV with date,value columns."""
    path = DATA_DIR / f"{name_or_path}.csv" if name_or_path in DATASETS else Path(name_or_path)
    if not path.exists():
        if name_or_path in DATASETS:  # regenerate if the bundled file was deleted
            return synthetic_series(name_or_path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "value" not in cols:
        raise ValueError("CSV needs 'date' and 'value' columns")
    df = df.rename(columns={cols["date"]: "date", cols["value"]: "value"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    # Reindex to a continuous daily calendar and fill small gaps by interpolation.
    full = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
    df = full.merge(df, on="date", how="left")
    df["value"] = df["value"].interpolate(limit_direction="both")
    for col in ("promo", "holiday"):
        if col in df:
            df[col] = df[col].fillna(0).astype(int)
    return df.reset_index(drop=True)
