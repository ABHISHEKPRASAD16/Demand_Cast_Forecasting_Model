"""Time-based train/test splitting.

Ordinary k-fold CV shuffles rows, which lets a model trained on next
week's data "predict" last week's — for a time series that's leakage,
not validation. Every split here respects chronological order: a fold's
training data always ends before its validation data begins.
"""

from collections.abc import Iterator
from datetime import timedelta

import pandas as pd


def train_test_split_by_date(
    df: pd.DataFrame, date_col: str, test_weeks: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the final `test_weeks` of calendar time as an untouched test set.

    Applying one shared cutoff date (rather than a per-row percentage) means
    every store in the panel is evaluated on the same forecast horizon, which
    matches how the model would actually be used in production.
    """
    cutoff = df[date_col].max() - timedelta(weeks=test_weeks) + timedelta(days=1)
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test


def expanding_window_splits(
    df: pd.DataFrame, date_col: str, n_splits: int = 3, val_weeks: int = 6
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling-origin CV folds, oldest to newest, for model/hyperparameter selection.

    Call this on the *training* portion returned by `train_test_split_by_date`,
    never on the full dataset — the final holdout must stay untouched until the
    one-time comparison-table evaluation. Each fold trains on all history before
    its validation window and validates on the next `val_weeks`, so a model that
    only looks good on one lucky window gets caught.
    """
    max_date = df[date_col].max()
    for k in range(n_splits, 0, -1):
        val_end = max_date - timedelta(weeks=(k - 1) * val_weeks)
        val_start = val_end - timedelta(weeks=val_weeks) + timedelta(days=1)
        train_fold = df[df[date_col] < val_start]
        val_fold = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)]
        if train_fold.empty or val_fold.empty:
            continue
        yield train_fold.copy(), val_fold.copy()
