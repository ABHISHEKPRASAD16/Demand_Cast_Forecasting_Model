import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from demandcast.serving.inference import (
    get_history_fetcher,
    get_model,
    get_model_metadata,
    get_model_version,
)
from demandcast.serving.main import app


class _StubModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([123.45] * len(X))


def _sufficient_history(store_id: int = 1, n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "store_id": store_id,
            "sale_date": dates,
            "day_of_week": dates.dayofweek + 1,
            "sales": range(100, 100 + n_days),
            "is_promo": False,
            "is_school_holiday": False,
            "state_holiday": "0",
            "competition_distance": 500.0,
            "has_promo2": False,
            "store_type": "a",
            "assortment": "a",
        }
    )


def _override_deps(history: pd.DataFrame):
    app.dependency_overrides[get_model] = lambda: _StubModel()
    app.dependency_overrides[get_model_version] = lambda: "1"
    app.dependency_overrides[get_history_fetcher] = lambda: (lambda store_id, before_date: history)


def _clear_overrides():
    app.dependency_overrides.clear()


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_for_valid_request():
    _override_deps(_sufficient_history())
    try:
        response = client.post(
            "/predict",
            json={"store_id": 1, "date": "2015-02-15", "is_promo": True, "state_holiday": "0"},
        )
    finally:
        _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_sales"] == 123.45
    assert body["store_id"] == 1
    assert body["model_version"] == "1"


def test_predict_returns_404_when_no_history_found():
    _override_deps(pd.DataFrame())
    try:
        response = client.post("/predict", json={"store_id": 999, "date": "2015-02-15"})
    finally:
        _clear_overrides()

    assert response.status_code == 404


def test_predict_returns_422_when_history_too_short():
    _override_deps(_sufficient_history(n_days=5))
    try:
        response = client.post("/predict", json={"store_id": 1, "date": "2015-02-15"})
    finally:
        _clear_overrides()

    assert response.status_code == 422


def test_predict_rejects_non_positive_store_id():
    response = client.post("/predict", json={"store_id": 0, "date": "2015-02-15"})
    assert response.status_code == 422


def test_model_metadata_returns_stubbed_info():
    app.dependency_overrides[get_model_metadata] = lambda: {
        "model_name": "demandcast-lightgbm",
        "model_version": "1",
        "run_id": "abc123",
        "metrics": {"wape": 8.75},
    }
    try:
        response = client.get("/model/metadata")
    finally:
        _clear_overrides()

    assert response.status_code == 200
    assert response.json()["metrics"]["wape"] == 8.75
