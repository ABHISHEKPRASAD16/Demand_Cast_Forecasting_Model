"""Model/data access for the API, kept separate from routing so tests can
swap in stubs via FastAPI's dependency_overrides instead of touching the
real DuckDB warehouse or MLflow registry.
"""

import logging
from collections.abc import Callable
from datetime import date
from functools import lru_cache

import mlflow
import pandas as pd

from demandcast.data import load_fct_sales
from demandcast.features.engineering import build_features_for_prediction
from demandcast.registry import MLFLOW_TRACKING_URI, get_latest_model_version, load_latest_model
from demandcast.serving.schemas import PredictRequest

logger = logging.getLogger(__name__)

HistoryFetcher = Callable[[int, date], pd.DataFrame]

# a bit more than MIN_HISTORY_DAYS so a handful of missing/closed days
# near the edge don't leave too little history to work with
LOOKBACK_DAYS = 45


@lru_cache(maxsize=1)
def _cached_model_and_version() -> tuple:
    version_info = get_latest_model_version()
    model = load_latest_model()
    return model, str(version_info.version)


def get_model():
    """FastAPI dependency: the current production LightGBM booster (loaded once)."""
    model, _ = _cached_model_and_version()
    return model


def get_model_version() -> str:
    """FastAPI dependency: the registry version of the currently loaded model."""
    _, version = _cached_model_and_version()
    return version


def get_store_history(store_id: int, before_date: date) -> pd.DataFrame:
    """Trailing daily history for one store, strictly before `before_date`."""
    df = load_fct_sales(store_id=store_id)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    history = df[df["sale_date"] < pd.Timestamp(before_date)].sort_values("sale_date")
    return history.tail(LOOKBACK_DAYS)


def get_history_fetcher() -> HistoryFetcher:
    """FastAPI dependency: swapped out in tests for a stub that doesn't hit DuckDB."""
    return get_store_history


def get_model_metadata() -> dict:
    """FastAPI dependency: current registered model's version + its training metrics."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version_info = get_latest_model_version()
    run = mlflow.get_run(version_info.run_id)
    return {
        "model_name": version_info.name,
        "model_version": str(version_info.version),
        "run_id": version_info.run_id,
        "metrics": dict(run.data.metrics),
    }


def predict_sales(model, history: pd.DataFrame, request: PredictRequest) -> float:
    features = build_features_for_prediction(
        history,
        target_date=request.date,
        is_promo=request.is_promo,
        is_school_holiday=request.is_school_holiday,
        state_holiday=request.state_holiday,
    )
    return float(model.predict(features)[0])
