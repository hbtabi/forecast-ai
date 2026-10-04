import numpy as np
import pandas as pd
import pytest

from forecast_ai.evaluate import backtest, cutoffs
from forecast_ai.metrics import all_metrics, bias, mae, mape, mase, rmse
from forecast_ai.models import FEATURE_NAMES, GradientBoosting, HoltWinters, SeasonalNaive, _row_features


def split(df, h=28):
    return df.iloc[:-h], df.iloc[-h:]


def test_metrics_on_known_values():
    y, p = np.array([100, 200, 300.0]), np.array([110, 190, 300.0])
    assert mae(y, p) == pytest.approx(20 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(200 / 3))
    assert mape(y, p) == pytest.approx((10 + 5 + 0) / 3)
    assert bias(y, p) == pytest.approx(0.0)
    hist = np.tile([1.0, 2.0], 10)
    assert mase(y, p, hist, season=1) == pytest.approx(mae(y, p) / 1.0)
    assert set(all_metrics(y, p, hist)) == {"MAE", "RMSE", "MAPE", "MASE", "Bias"}


def test_seasonal_naive_repeats_last_week(sales):
    train, test = split(sales)
    pred = SeasonalNaive().fit(train).predict(test)
    assert np.allclose(pred[:7], train["value"].to_numpy()[-7:])
    assert np.allclose(pred[7:14], pred[:7])


def test_holt_winters_recovers_a_clean_seasonal_pattern():
    t = np.arange(140)
    weekly = np.array([1.0, 1.0, 1.1, 1.2, 1.4, 1.8, 1.3])
    y = 500 * (1 + 0.002 * t) * weekly[t % 7]
    df = pd.DataFrame({"date": pd.date_range("2026-01-05", periods=140), "value": y})
    train, test = split(df)
    pred = HoltWinters().fit(train).predict(test)
    assert mape(test["value"], pred) < 2.0
    params = HoltWinters().fit(train).params()
    assert 0 < params["alpha"] < 1 and 0.8 <= params["phi"] <= 0.999


def test_holt_winters_needs_enough_history():
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "value": np.ones(10)})
    with pytest.raises(ValueError):
        HoltWinters().fit(df)


def test_feature_row_matches_names(sales):
    values = sales["value"].to_numpy()
    row = _row_features(values, 400, sales["date"].iloc[400], 1, 0)
    assert len(row) == len(FEATURE_NAMES)
    assert np.isnan(_row_features(values, 100, sales["date"].iloc[100], 0, 0)[FEATURE_NAMES.index("lag_364")])


def test_gradient_boosting_beats_baseline_and_uses_promos(sales):
    train, test = split(sales)
    gbm = GradientBoosting(max_iter=150).fit(train)
    pred = gbm.predict(test.drop(columns="value"))
    base = SeasonalNaive().fit(train).predict(test)
    assert pred.shape == (28,) and np.isfinite(pred).all()
    assert mae(test["value"], pred) < mae(test["value"], base)
    # Turning a promotion on should lift that day's forecast.
    fut = test.drop(columns="value").assign(promo=0)
    lifted = fut.copy()
    lifted.loc[lifted.index[3], "promo"] = 1
    assert gbm.predict(lifted)[3] > gbm.predict(fut)[3]


def test_backtest_produces_ranked_table(sales):
    assert cutoffs(500, 28, 3) == [472, 444, 416]
    comp = backtest(sales, horizon=14, folds=2, models=[SeasonalNaive(), HoltWinters()], dataset="sales")
    table = comp.table()
    assert list(table.columns) == ["MAE", "RMSE", "MAPE", "MASE", "Bias", "fit_s"]
    assert table["MASE"].is_monotonic_increasing
    assert comp.best_model() == table.index[0]
    assert set(comp.fold_forecasts(0)) == {"Seasonal naive", "Holt-Winters"}
