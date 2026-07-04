import numpy as np

from demandcast.models.baselines import (
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)


def test_naive_forecast_repeats_last_value():
    train = [1, 2, 3, 10]
    result = naive_forecast(train, horizon=5)
    assert np.array_equal(result, [10, 10, 10, 10, 10])


def test_seasonal_naive_repeats_last_season_cyclically():
    train = list(range(1, 15))  # last 7 = [8..14]
    result = seasonal_naive_forecast(train, horizon=10, season_length=7)
    expected = [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]
    assert np.array_equal(result, expected)


def test_moving_average_forecast_is_flat_at_the_mean():
    train = [10, 20, 30, 40]
    result = moving_average_forecast(train, horizon=3, window=4)
    assert np.allclose(result, [25, 25, 25])


def test_moving_average_uses_only_the_trailing_window():
    train = [1000, 1000, 10, 20, 30]
    result = moving_average_forecast(train, horizon=1, window=3)
    assert np.allclose(result, [20])
