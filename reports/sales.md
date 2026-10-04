# Forecast report: sales

_Generated 04 Oct 2026 20:18 by forecast-ai · 1,095 days of data (01 Oct 2023 to 29 Sep 2026) · horizon 28 days · 3 models × 3 backtest folds_

## Summary

**Gradient boosting** is the most accurate model for sales over a 28-day horizon, with an average error of **6.3%** (MASE 0.60). That is **50% lower error** than the seasonal-naive baseline ("same day last week", 13.0% error). On average it under-forecasts by 0.6%, so treat the forecast as close to unbiased. Holt-Winters (8.7%) is a fast, transparent fallback that needs no feature engineering.

<sub>Summary written by: `template`</sub>

## Backtest results (mean across folds)

| model | MAE | RMSE | MAPE | MASE | Bias | fit_s |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient boosting** | **108.8** | **145.0** | **6.31%** | **0.598** | **-0.57%** | **1.31s** |
| Holt-Winters | 149.7 | 191.6 | 8.72% | 0.823 | -1.21% | 0.16s |
| Seasonal naive | 216.4 | 276.1 | 12.97% | 1.190 | +2.23% | 0.00s |

MAPE = mean absolute % error · MASE < 1 beats "same day last week" · Bias > 0 = over-forecasting

## Holdout forecast

![Holdout forecast](../docs/sales_forecast.png)

## Backtest accuracy

![Backtest accuracy](../docs/sales_backtest.png)

## Feature importance

![Feature importance](../docs/sales_importance.png)

## Model notes

- Holt-Winters fitted smoothing parameters: alpha = 0.135, beta = 0.000, gamma = 0.057, phi = 0.978
- Gradient boosting uses lags 1-28, a 7-day mean, same-day-last-year and calendar/promo/holiday features, all relative to the trailing 28-day mean so it can follow growth.
