"""Build the Phase 2 model comparison table.

Runs every model family against the *same* held-out test window (the final
`test_weeks` of calendar time) and writes results to reports/. Classical
models (baselines, ARIMA, Prophet) are fit on one representative store;
LightGBM and the LSTM are trained globally across all stores, and are also
reported on that same representative store's subset of rows so every model
can be compared on identical ground at least once.

This script itself is the "notebook-free" driver Phase 3 will later wrap in
a config-driven training pipeline - for now it's a plain, rerunnable script
because MLflow / experiment tracking doesn't exist yet.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from demandcast.config import load_config
from demandcast.data import load_fct_sales
from demandcast.evaluation.metrics import all_metrics
from demandcast.evaluation.splits import expanding_window_splits, train_test_split_by_date
from demandcast.features.engineering import build_features
from demandcast.models.baselines import (
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)
from demandcast.models.deep import (
    SlidingWindowDataset,
    build_store_index,
    fit_store_scalers,
    predict_lstm,
    train_lstm,
)
from demandcast.models.ml import predict_lightgbm, train_lightgbm
from demandcast.models.statistical import (
    fit_prophet,
    fit_sarimax,
    prophet_forecast,
    sarimax_forecast,
    select_sarimax_order,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parents[3] / "reports"


def _row(model: str, scope: str, y_true, y_pred) -> dict:
    return {"model": model, "scope": scope, **all_metrics(y_true, y_pred)}


def run_baselines_and_statistical(store_id: int, test_weeks: int) -> list[dict]:
    scope = f"Store {store_id}"
    df = load_fct_sales(store_id=store_id)
    train, test = train_test_split_by_date(df, "sale_date", test_weeks=test_weeks)
    horizon = len(test)
    results = []

    logger.info("[%s] baselines", scope)
    results.append(_row("Naive", scope, test["sales"], naive_forecast(train["sales"], horizon)))
    results.append(
        _row(
            "Seasonal naive (7d)",
            scope,
            test["sales"],
            seasonal_naive_forecast(train["sales"], horizon),
        )
    )
    results.append(
        _row(
            "Moving average (7d)",
            scope,
            test["sales"],
            moving_average_forecast(train["sales"], horizon),
        )
    )

    logger.info("[%s] SARIMAX order search", scope)
    order = select_sarimax_order(train["sales"])
    logger.info("[%s] SARIMAX best order: %s", scope, order)
    arima_result = fit_sarimax(train["sales"], order)
    results.append(
        _row(f"SARIMAX{order}", scope, test["sales"], sarimax_forecast(arima_result, horizon))
    )

    logger.info("[%s] Prophet", scope)
    prophet_model = fit_prophet(train[["sale_date", "sales", "is_promo"]])
    prophet_preds = prophet_forecast(prophet_model, test[["sale_date", "is_promo"]])
    results.append(_row("Prophet", scope, test["sales"], prophet_preds))

    return results


def run_lightgbm(test_weeks: int, cv_val_weeks: int, representative_store_id: int) -> list[dict]:
    logger.info("[Global] loading all-store data for LightGBM")
    df = load_fct_sales()
    features = build_features(df)

    train_all, test = train_test_split_by_date(features.df, "sale_date", test_weeks=test_weeks)
    *_, (train, val) = expanding_window_splits(
        train_all, "sale_date", n_splits=1, val_weeks=cv_val_weeks
    )

    logger.info("[Global] training LightGBM on %d rows", len(train))
    booster = train_lightgbm(
        train, features.feature_columns, features.categorical_columns, val_df=val
    )
    preds = predict_lightgbm(booster, test, features.feature_columns)

    results = [_row("LightGBM", "Global (1,115 stores)", test["sales"], preds)]

    store_mask = test["store_id"] == str(representative_store_id)
    if store_mask.any():
        results.append(
            _row(
                "LightGBM",
                f"Store {representative_store_id}",
                test.loc[store_mask, "sales"],
                preds[store_mask.to_numpy()],
            )
        )
    return results


def run_lstm(test_weeks: int, cv_val_weeks: int, representative_store_id: int) -> list[dict]:
    logger.info("[Global] loading all-store data for LSTM")
    df = load_fct_sales()
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    train_all, test = train_test_split_by_date(df, "sale_date", test_weeks=test_weeks)
    *_, (train, val) = expanding_window_splits(
        train_all, "sale_date", n_splits=1, val_weeks=cv_val_weeks
    )

    index = build_store_index(df)
    scalers = fit_store_scalers(train)

    train_ds = SlidingWindowDataset(
        df, index, scalers, train["sale_date"].min(), train["sale_date"].max()
    )
    val_ds = SlidingWindowDataset(
        df, index, scalers, val["sale_date"].min(), val["sale_date"].max()
    )
    test_ds = SlidingWindowDataset(
        df, index, scalers, test["sale_date"].min(), test["sale_date"].max()
    )

    logger.info("[Global] training LSTM on %d windows (%d stores)", len(train_ds), index.num_stores)
    torch.manual_seed(0)
    t0 = time.time()
    model, history = train_lstm(
        train_ds, val_ds, index.num_stores, epochs=15, patience=3, batch_size=512
    )
    logger.info(
        "[Global] LSTM trained in %.1fs, %d epochs run", time.time() - t0, len(history.val_loss)
    )

    preds = predict_lstm(model, test_ds)
    actuals = np.array(test_ds.targets_actual)
    results = [_row("LSTM", "Global (1,115 stores)", actuals, preds)]

    store_mask = np.array(test_ds.store_ids) == representative_store_id
    if store_mask.any():
        results.append(
            _row("LSTM", f"Store {representative_store_id}", actuals[store_mask], preds[store_mask])
        )
    return results


def main() -> None:
    config = load_config()
    m = config.modeling

    all_results = []
    all_results += run_baselines_and_statistical(m.representative_store_id, m.test_weeks)
    all_results += run_lightgbm(m.test_weeks, m.cv_val_weeks, m.representative_store_id)
    all_results += run_lstm(m.test_weeks, m.cv_val_weeks, m.representative_store_id)

    results_df = pd.DataFrame(all_results)[["model", "scope", "mape", "rmse", "wape"]]
    results_df = results_df.sort_values("scope", kind="stable")

    REPORTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    (REPORTS_DIR / "model_comparison.md").write_text(
        results_df.to_markdown(index=False, floatfmt=".2f")
    )

    logger.info("\n%s", results_df.to_string(index=False))
    logger.info("Wrote reports/model_comparison.csv and reports/model_comparison.md")


if __name__ == "__main__":
    main()
