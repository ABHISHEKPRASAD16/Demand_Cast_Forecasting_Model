import numpy as np
import pandas as pd

from demandcast.models.statistical import (
    fit_prophet,
    fit_sarimax,
    prophet_forecast,
    sarimax_forecast,
    select_sarimax_order,
)


def _synthetic_series(n_days: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    days = np.arange(n_days)
    weekly = 10 * np.sin(2 * np.pi * days / 7)
    return 100 + weekly + rng.normal(0, 1, n_days)


def test_select_sarimax_order_returns_a_valid_order():
    series = _synthetic_series()
    order = select_sarimax_order(series, pq_range=range(1))
    assert order == (0, 1, 0)


def test_sarimax_forecast_returns_requested_horizon():
    series = _synthetic_series()
    result = fit_sarimax(series, order=(0, 1, 0))
    forecast = sarimax_forecast(result, horizon=5)
    assert forecast.shape == (5,)
    assert np.all(np.isfinite(forecast))


def _synthetic_df(n_days: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sale_date": pd.date_range("2015-01-01", periods=n_days, freq="D"),
            "sales": _synthetic_series(n_days),
            "is_promo": [i % 3 == 0 for i in range(n_days)],
        }
    )


def test_prophet_forecast_returns_requested_horizon():
    train_df = _synthetic_df(40)
    future_df = pd.DataFrame(
        {
            "sale_date": pd.date_range("2015-02-10", periods=5, freq="D"),
            "is_promo": [True, False, False, True, False],
        }
    )

    model = fit_prophet(train_df)
    forecast = prophet_forecast(model, future_df)

    assert forecast.shape == (5,)
    assert np.all(np.isfinite(forecast))
