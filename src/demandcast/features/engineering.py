"""Feature engineering shared by the global LightGBM and LSTM models.

Everything here is computed with `shift`/lookback windows so that the
features available for a given store-date only use information a real
forecast run would have had at that point in time. `customers` and
`is_open` are deliberately excluded from the feature set even though
they're in fct_sales: `customers` is itself an outcome correlated with
sales that we wouldn't know in advance, and `is_open` is constant True
here because the loader already filters to open days.
"""

from dataclasses import dataclass, field

import pandas as pd

LAG_DAYS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)
CATEGORICAL_COLUMNS = ("store_id", "store_type", "assortment", "state_holiday")


@dataclass
class FeatureSet:
    """A feature-engineered frame plus the column groupings models need."""

    df: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str] = field(default_factory=lambda: list(CATEGORICAL_COLUMNS))
    target_column: str = "sales"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features derived purely from the date/day_of_week already in fct_sales.

    Public (unlike the lag/rolling helpers below) because the LSTM module reuses
    it directly - a recurrent model should learn temporal dependencies from the
    raw sequence itself rather than being handed lag/rolling summaries.
    """
    df["month"] = df["sale_date"].dt.month
    df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)
    return df


def _add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "sale_date"])
    grouped_sales = df.groupby("store_id")["sales"]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grouped_sales.shift(lag)

    # shift(1) before rolling so today's own sales value never leaks into its own features
    shifted = grouped_sales.shift(1)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = shifted.groupby(df["store_id"]).transform(
            lambda s, window=window: s.rolling(window, min_periods=window).mean()
        )
        df[f"rolling_std_{window}"] = shifted.groupby(df["store_id"]).transform(
            lambda s, window=window: s.rolling(window, min_periods=window).std()
        )
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).astype("category")
    return df


def build_features(df: pd.DataFrame) -> FeatureSet:
    """Add calendar/lag/rolling/categorical features, ready for LightGBM or the LSTM.

    Drops the leading rows per store that don't yet have `max(LAG_DAYS)` days
    of history to compute lags from - there's no valid feature vector for
    them, so keeping them would just inject NaNs into training.
    """
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    df = add_calendar_features(df)
    df = _add_lag_and_rolling_features(df)
    df = _encode_categoricals(df)

    lag_rolling_cols = [f"lag_{lag}" for lag in LAG_DAYS] + [
        f"rolling_{stat}_{window}" for window in ROLLING_WINDOWS for stat in ("mean", "std")
    ]
    df = df.dropna(subset=lag_rolling_cols).reset_index(drop=True)

    feature_columns = [
        "day_of_week",
        "month",
        "week_of_year",
        "is_weekend",
        "is_promo",
        "is_school_holiday",
        "competition_distance",
        "has_promo2",
        *lag_rolling_cols,
        *CATEGORICAL_COLUMNS,
    ]
    return FeatureSet(df=df, feature_columns=feature_columns)
