"""Config-driven training entrypoint: validate -> build features/dataset ->
train -> evaluate -> log everything to MLflow -> (optionally) register.

    python -m demandcast.training.train --config config/models/lightgbm.yaml

Every run is versioned and comparable in the MLflow UI (`mlflow ui
--backend-store-uri sqlite:///mlflow.db`), instead of numbers only living in
a printed table like Phase 2's run_comparison.py.
"""

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from demandcast.data import load_fct_sales
from demandcast.evaluation.metrics import all_metrics
from demandcast.evaluation.splits import expanding_window_splits, train_test_split_by_date
from demandcast.features.engineering import build_features
from demandcast.models.deep import (
    SlidingWindowDataset,
    build_store_index,
    fit_store_scalers,
    predict_lstm,
    train_lstm,
)
from demandcast.models.ml import predict_lightgbm, train_lightgbm
from demandcast.training.config import (
    LightGBMTrainingConfig,
    LSTMTrainingConfig,
    TrainingConfig,
    load_training_config,
)
from demandcast.validation import validate_fct_sales_or_raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "demandcast"


def run_lightgbm(cfg: LightGBMTrainingConfig) -> None:
    df = load_fct_sales()
    validate_fct_sales_or_raise(df)

    features = build_features(df)
    train_all, test = train_test_split_by_date(features.df, "sale_date", test_weeks=cfg.test_weeks)
    *_, (train, val) = expanding_window_splits(
        train_all, "sale_date", n_splits=1, val_weeks=cfg.cv_val_weeks
    )

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params(
            {
                "model_type": "lightgbm",
                "num_boost_round": cfg.num_boost_round,
                "early_stopping_rounds": cfg.early_stopping_rounds,
                **cfg.params,
            }
        )

        booster = train_lightgbm(
            train,
            features.feature_columns,
            features.categorical_columns,
            val_df=val,
            params=cfg.params,
            num_boost_round=cfg.num_boost_round,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        mlflow.log_metric("best_iteration", booster.best_iteration)

        preds = predict_lightgbm(booster, test, features.feature_columns)
        metrics = all_metrics(test["sales"], preds)
        mlflow.log_metrics(metrics)
        logger.info("LightGBM test metrics: %s", metrics)

        model_info = mlflow.lightgbm.log_model(booster, name="model")
        if cfg.register_model:
            mlflow.register_model(model_info.model_uri, "demandcast-lightgbm")
            logger.info("Registered model as demandcast-lightgbm")


def run_lstm(cfg: LSTMTrainingConfig) -> None:
    df = load_fct_sales()
    validate_fct_sales_or_raise(df)
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    train_all, test = train_test_split_by_date(df, "sale_date", test_weeks=cfg.test_weeks)
    *_, (train, val) = expanding_window_splits(
        train_all, "sale_date", n_splits=1, val_weeks=cfg.cv_val_weeks
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

    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params(
            {
                "model_type": "lstm",
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "patience": cfg.patience,
                "embedding_dim": cfg.embedding_dim,
                "hidden_size": cfg.hidden_size,
            }
        )

        model, history = train_lstm(
            train_ds,
            val_ds,
            index.num_stores,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            patience=cfg.patience,
            embedding_dim=cfg.embedding_dim,
            hidden_size=cfg.hidden_size,
        )
        for epoch, (train_loss, val_loss) in enumerate(
            zip(history.train_loss, history.val_loss, strict=True)
        ):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        preds = predict_lstm(model, test_ds)
        metrics = all_metrics(np.array(test_ds.targets_actual), preds)
        mlflow.log_metrics(metrics)
        logger.info("LSTM test metrics: %s", metrics)

        # pt2 (mlflow's new default) traces the forward graph from a concrete
        # input example; this model takes three positional tensor args, so
        # the classic pickle format is the simpler fit here.
        model_info = mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
        if cfg.register_model:
            mlflow.register_model(model_info.model_uri, "demandcast-lstm")
            logger.info("Registered model as demandcast-lstm")


def run(cfg: TrainingConfig) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    if isinstance(cfg, LightGBMTrainingConfig):
        run_lightgbm(cfg)
    elif isinstance(cfg, LSTMTrainingConfig):
        run_lstm(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    run(load_training_config(args.config))


if __name__ == "__main__":
    main()
