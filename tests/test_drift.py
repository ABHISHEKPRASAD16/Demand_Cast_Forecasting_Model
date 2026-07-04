import numpy as np
import pandas as pd

from demandcast.monitoring.drift import generate_drift_report

FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "is_weekend",
    "is_promo",
    "competition_distance",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
    "store_type",
    "assortment",
]


class _StubModel:
    """Predicts the 7-day rolling mean - enough behavior to exercise
    prediction/target drift without needing a real trained booster."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["rolling_mean_7"].to_numpy()


def _synthetic_frame(n_rows: int, sales_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "day_of_week": rng.integers(1, 8, n_rows),
            "month": rng.integers(1, 13, n_rows),
            "is_weekend": rng.integers(0, 2, n_rows),
            "is_promo": rng.integers(0, 2, n_rows),
            "competition_distance": rng.uniform(100, 5000, n_rows),
            "lag_7": rng.normal(100 + sales_shift, 10, n_rows),
            "lag_14": rng.normal(100 + sales_shift, 10, n_rows),
            "lag_28": rng.normal(100 + sales_shift, 10, n_rows),
            "rolling_mean_7": rng.normal(100 + sales_shift, 10, n_rows),
            "rolling_std_7": rng.uniform(1, 5, n_rows),
            "rolling_mean_28": rng.normal(100 + sales_shift, 10, n_rows),
            "rolling_std_28": rng.uniform(1, 5, n_rows),
            "store_type": rng.choice(["a", "b", "c", "d"], n_rows),
            "assortment": rng.choice(["a", "b", "c"], n_rows),
            "sales": rng.normal(100 + sales_shift, 10, n_rows),
        }
    )


def test_generate_drift_report_runs_on_shifted_data(tmp_path):
    reference = _synthetic_frame(200)
    current = _synthetic_frame(200, sales_shift=50.0)  # simulate a real distribution shift

    run = generate_drift_report(reference, current, FEATURE_COLUMNS, _StubModel())

    output_path = tmp_path / "drift_report.html"
    run.save_html(str(output_path))
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_drift_report_does_not_mutate_inputs():
    reference = _synthetic_frame(50)
    current = _synthetic_frame(50)
    reference_cols_before = list(reference.columns)

    generate_drift_report(reference, current, FEATURE_COLUMNS, _StubModel())

    assert list(reference.columns) == reference_cols_before
    assert "prediction" not in reference.columns
