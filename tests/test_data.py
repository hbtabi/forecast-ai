from datetime import date

import pandas as pd
import pytest

from forecast_ai.data import _easter, load_dataset, synthetic_series, uk_bank_holidays


def test_easter_and_bank_holidays():
    assert _easter(2024) == date(2024, 3, 31)
    assert _easter(2026) == date(2026, 4, 5)
    hols = uk_bank_holidays([2026])
    assert date(2026, 4, 3) in hols          # Good Friday
    assert date(2026, 5, 25) in hols         # Spring bank holiday (last Monday of May)
    assert date(2026, 8, 31) in hols         # Summer bank holiday
    assert len(hols) == 8


def test_synthetic_series_has_realistic_patterns():
    df = synthetic_series("footfall", days=730)
    assert list(df.columns) == ["date", "value", "promo", "holiday"]
    assert (df["value"] > 0).all()
    by_dow = df.groupby(df["date"].dt.dayofweek)["value"].mean()
    assert by_dow.idxmax() == 5                       # Saturday is busiest for footfall
    assert synthetic_series("footfall", days=50).equals(synthetic_series("footfall", days=50))  # deterministic
    with pytest.raises(ValueError):
        synthetic_series("weather")


def test_bundled_datasets_load():
    for name in ("sales", "footfall"):
        df = load_dataset(name)
        assert len(df) >= 1000
        assert df["date"].is_monotonic_increasing


def test_custom_csv_is_cleaned(tmp_path):
    p = tmp_path / "my.csv"
    raw = pd.DataFrame({"Date": ["2026-01-03", "2026-01-01", "2026-01-01", "2026-01-05"], "Value": [30, 10, 10, 50]})
    raw.to_csv(p, index=False)
    df = load_dataset(str(p))
    assert df["date"].dt.strftime("%d").tolist() == ["01", "02", "03", "04", "05"]   # sorted, deduped, gaps filled
    assert df["value"].tolist() == [10, 20, 30, 40, 50]                               # linear interpolation
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        load_dataset(str(bad))
    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path / "missing.csv"))
