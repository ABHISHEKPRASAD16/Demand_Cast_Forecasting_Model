"""Drift report: does the data the model was trained on still look like
what it's now being asked to predict on?

Scores both windows with the *latest registered* demandcast-lightgbm model
pulled from the MLflow Model Registry, rather than retraining - that mirrors
how this would actually run in production: a scheduled job that evaluates
the deployed model version against incoming data, on some cadence, without
touching training.

Reference = the training window, current = the test window - the same
split every other Phase 2/3 script uses. In real operation "current" would
instead be last week's incoming batch; using the test window here is a
stand-in that still answers the real question: would this job have caught
the shift between what the model learned from and what it was evaluated on?
"""

import logging
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.presets import DataDriftPreset, RegressionPreset

from demandcast.data import load_fct_sales
from demandcast.evaluation.splits import train_test_split_by_date
from demandcast.features.engineering import FeatureSet, build_features
from demandcast.registry import load_latest_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[3] / "reports" / "drift"
REFERENCE_SAMPLE_SIZE = 50_000

NUMERIC_DRIFT_COLUMNS = [
    "competition_distance",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
]
CATEGORICAL_DRIFT_COLUMNS = [
    "day_of_week",
    "month",
    "is_weekend",
    "is_promo",
    "store_type",
    "assortment",
]


def build_reference_and_current(
    test_weeks: int = 6, reference_sample_size: int = REFERENCE_SAMPLE_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureSet]:
    df = load_fct_sales()
    features = build_features(df)
    train, test = train_test_split_by_date(features.df, "sale_date", test_weeks=test_weeks)

    reference = train.sample(n=min(reference_sample_size, len(train)), random_state=42)
    return reference, test, features


def generate_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, feature_columns: list[str], model
) -> Report:
    reference = reference.copy()
    current = current.copy()
    reference["prediction"] = model.predict(reference[feature_columns])
    current["prediction"] = model.predict(current[feature_columns])
    reference = reference.rename(columns={"sales": "target"})
    current = current.rename(columns={"sales": "target"})

    definition = DataDefinition(
        numerical_columns=NUMERIC_DRIFT_COLUMNS,
        categorical_columns=CATEGORICAL_DRIFT_COLUMNS,
        regression=[Regression()],
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
    current_dataset = Dataset.from_pandas(current, data_definition=definition)

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    return report.run(reference_data=reference_dataset, current_data=current_dataset)


def main() -> None:
    reference, current, features = build_reference_and_current()
    model = load_latest_model()

    run = generate_drift_report(reference, current, features.feature_columns, model)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "drift_report.html"
    run.save_html(str(output_path))
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
