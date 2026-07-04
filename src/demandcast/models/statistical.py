"""Classical time-series models: seasonal ARIMA (SARIMAX) and Prophet.

Both are fit on a single representative store rather than globally - unlike
LightGBM/the LSTM, these model one series at a time, so "training" 1,115 of
them would multiply runtime without teaching anything new about the method.
"""

import logging
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# Daily retail sales have strong week-over-week seasonality; seasonal
# differencing (D=1) at a 7-day period removes it, and a single seasonal
# AR + MA term (P=1, Q=1) captures what's left. These are fixed rather than
# grid-searched to keep the non-seasonal grid search below tractable.
SEASONAL_ORDER = (1, 1, 1, 7)


def select_sarimax_order(
    train_sales: npt.ArrayLike, pq_range: range = range(3), d: int = 1
) -> tuple[int, int, int]:
    """Grid-search non-seasonal (p, d, q) by AIC - the standard, cheaper stand-in
    for auto_arima when that package isn't part of the stack."""
    train_sales = np.asarray(train_sales, dtype=float)
    best_order, best_aic = (0, d, 0), np.inf

    for p in pq_range:
        for q in pq_range:
            order = (p, d, q)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = SARIMAX(
                        train_sales,
                        order=order,
                        seasonal_order=SEASONAL_ORDER,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)
            except Exception:  # noqa: BLE001 - some orders fail to converge; skip them
                continue
            if result.aic < best_aic:
                best_order, best_aic = order, result.aic

    return best_order


def fit_sarimax(train_sales: npt.ArrayLike, order: tuple[int, int, int]) -> SARIMAXResultsWrapper:
    train_sales = np.asarray(train_sales, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(
            train_sales,
            order=order,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)


def sarimax_forecast(result: SARIMAXResultsWrapper, horizon: int) -> np.ndarray:
    return np.asarray(result.get_forecast(steps=horizon).predicted_mean)


def fit_prophet(train_df: pd.DataFrame) -> Prophet:
    """Fit Prophet with weekly/yearly seasonality plus promo as a known-in-advance regressor."""
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.add_regressor("is_promo")

    prophet_df = train_df.rename(columns={"sale_date": "ds", "sales": "y"})[["ds", "y", "is_promo"]]
    prophet_df["is_promo"] = prophet_df["is_promo"].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)
    return model


def prophet_forecast(model: Prophet, future_df: pd.DataFrame) -> np.ndarray:
    """future_df must have `sale_date` and `is_promo` for the forecast horizon."""
    future = future_df.rename(columns={"sale_date": "ds"})[["ds", "is_promo"]].copy()
    future["is_promo"] = future["is_promo"].astype(int)
    forecast = model.predict(future)
    return forecast["yhat"].to_numpy()
