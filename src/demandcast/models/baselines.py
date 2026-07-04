"""Baseline forecasters: the floor every fancier model has to beat.

If ARIMA or LightGBM can't clear seasonal-naive, they're not adding value -
a 7-day lag is a real competitor for daily retail sales precisely because
weekly seasonality (weekday effects, weekend closures) explains most of the
variance on its own.
"""

import numpy as np
import numpy.typing as npt


def naive_forecast(train_sales: npt.ArrayLike, horizon: int) -> np.ndarray:
    """Repeat the last observed value for every step of the horizon."""
    train_sales = np.asarray(train_sales, dtype=float)
    return np.full(horizon, train_sales[-1])


def seasonal_naive_forecast(
    train_sales: npt.ArrayLike, horizon: int, season_length: int = 7
) -> np.ndarray:
    """Repeat the last full season (default: last 7 days) cyclically across the horizon."""
    train_sales = np.asarray(train_sales, dtype=float)
    last_season = train_sales[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]


def moving_average_forecast(
    train_sales: npt.ArrayLike, horizon: int, window: int = 7
) -> np.ndarray:
    """Forecast a flat line at the mean of the last `window` observations."""
    train_sales = np.asarray(train_sales, dtype=float)
    mean_value = train_sales[-window:].mean()
    return np.full(horizon, mean_value)
