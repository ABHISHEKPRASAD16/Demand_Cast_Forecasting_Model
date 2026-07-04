"""Forecast accuracy metrics shared across every model family.

MAPE is included because it's the metric stakeholders usually ask for by
name, but it is undefined at zero actuals and overweights low-volume
series/days relative to high-volume ones. WAPE (sum of errors over sum of
actuals) fixes both problems and is what actually drives the model
comparison table, so treat MAPE as a secondary, reported-for-familiarity
number rather than the one used to pick a winner.
"""

import numpy as np
import numpy.typing as npt


def _as_array(values: npt.ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def mape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Mean Absolute Percentage Error, in percent. Rows where y_true == 0 are dropped."""
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    mask = y_true != 0
    if not mask.any():
        raise ValueError("mape is undefined: all y_true values are zero")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Root Mean Squared Error, in the same units as y_true."""
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Weighted Absolute Percentage Error, in percent: sum(|error|) / sum(|actual|)."""
    y_true, y_pred = _as_array(y_true), _as_array(y_pred)
    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        raise ValueError("wape is undefined: y_true sums to zero")
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


def all_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """Convenience bundle used to build the model comparison table."""
    return {
        "mape": mape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }
