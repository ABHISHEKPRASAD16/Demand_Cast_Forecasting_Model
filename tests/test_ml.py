import numpy as np
import pandas as pd

from demandcast.features.engineering import build_features
from demandcast.models.ml import predict_lightgbm, train_lightgbm


def _synthetic_panel(n_days: int = 150, store_ids=(1, 2, 3)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for store_id in store_ids:
        dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
        day_of_week = dates.dayofweek.to_numpy() + 1
        base = 50 * store_id + 20 * np.sin(2 * np.pi * day_of_week / 7)
        sales = base + rng.normal(0, 1, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "store_id": store_id,
                    "sale_date": dates,
                    "day_of_week": day_of_week,
                    "sales": sales,
                    "is_promo": False,
                    "is_school_holiday": False,
                    "state_holiday": "0",
                    "competition_distance": 500.0,
                    "has_promo2": False,
                    "store_type": "a",
                    "assortment": "a",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_lightgbm_learns_the_weekly_pattern_better_than_the_mean():
    features = build_features(_synthetic_panel())
    df = features.df
    cutoff = df["sale_date"].quantile(0.8)
    train_df, test_df = df[df["sale_date"] < cutoff], df[df["sale_date"] >= cutoff]

    booster = train_lightgbm(
        train_df, features.feature_columns, features.categorical_columns, num_boost_round=100
    )
    preds = predict_lightgbm(booster, test_df, features.feature_columns)

    naive_rmse = np.sqrt(np.mean((test_df["sales"] - train_df["sales"].mean()) ** 2))
    model_rmse = np.sqrt(np.mean((test_df["sales"] - preds) ** 2))
    assert model_rmse < naive_rmse


def test_train_lightgbm_with_validation_set_uses_early_stopping():
    features = build_features(_synthetic_panel())
    df = features.df
    train_cutoff = df["sale_date"].quantile(0.6)
    val_cutoff = df["sale_date"].quantile(0.8)
    train_df = df[df["sale_date"] < train_cutoff]
    val_df = df[(df["sale_date"] >= train_cutoff) & (df["sale_date"] < val_cutoff)]

    booster = train_lightgbm(
        train_df,
        features.feature_columns,
        features.categorical_columns,
        val_df=val_df,
        num_boost_round=500,
        early_stopping_rounds=10,
    )

    assert 0 < booster.best_iteration < 500
