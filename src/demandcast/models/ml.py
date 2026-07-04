"""A single global LightGBM model trained across all stores.

One model with store_id, store_type, etc. as features generalizes better
than 1,115 tiny per-store models - low-volume stores borrow statistical
strength from similar ones, and it's what you'd actually deploy, since
retraining/serving 1,115 models is an operational burden a single global
model avoids entirely.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}


def train_lightgbm(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    target_column: str = "sales",
    val_df: pd.DataFrame | None = None,
    params: dict | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """Train on `train_df`. If `val_df` is given (an expanding-window CV fold
    carved out of the training period - never the final test holdout), use it
    for early stopping instead of a fixed, guessed num_boost_round."""
    resolved_params = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(
        train_df[feature_columns],
        label=train_df[target_column],
        categorical_feature=categorical_columns,
        free_raw_data=False,
    )

    valid_sets = [train_set]
    callbacks = [lgb.log_evaluation(period=0)]
    if val_df is not None:
        val_set = lgb.Dataset(
            val_df[feature_columns],
            label=val_df[target_column],
            categorical_feature=categorical_columns,
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets.append(val_set)
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    return lgb.train(
        resolved_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def predict_lightgbm(
    booster: lgb.Booster, df: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    num_iteration = booster.best_iteration if booster.best_iteration > 0 else None
    return booster.predict(df[feature_columns], num_iteration=num_iteration)
