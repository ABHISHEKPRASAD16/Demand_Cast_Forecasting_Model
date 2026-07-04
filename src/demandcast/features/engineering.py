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

LAG_ROLLING_COLUMNS = [f"lag_{lag}" for lag in LAG_DAYS] + [
    f"rolling_{stat}_{window}" for window in ROLLING_WINDOWS for stat in ("mean", "std")
]
FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "is_promo",
    "is_school_holiday",
    "competition_distance",
    "has_promo2",
    *LAG_ROLLING_COLUMNS,
    *CATEGORICAL_COLUMNS,
]
# max lookback needed before lag/rolling features are all defined for a row
MIN_HISTORY_DAYS = max(max(LAG_DAYS), max(ROLLING_WINDOWS))


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
    df = df.dropna(subset=LAG_ROLLING_COLUMNS).reset_index(drop=True)

    return FeatureSet(df=df, feature_columns=list(FEATURE_COLUMNS))


def build_features_for_prediction(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
    is_promo: bool,
    is_school_holiday: bool,
    state_holiday: str = "0",
) -> pd.DataFrame:
    """Build the one-row feature vector for a live (store, target_date) prediction.

    `history` must be that store's daily rows strictly before target_date,
    with at least MIN_HISTORY_DAYS of them - the same lag/rolling logic
    build_features uses at training time, just run on a single store's
    trailing window instead of the whole panel. The synthetic target row's
    own `sales` value is never used: lag/rolling features come from
    shift()ed history, not the row itself, so a placeholder is fine.

    Categorical columns are re-encoded from whatever values are present in
    this small window - that's safe because LightGBM's Booster carries its
    own training-time category mapping (`pandas_categorical`) and re-applies
    it at predict time, regardless of which categories a given input frame's
    dtype happens to include.
    """
    history = history.copy()
    history["sale_date"] = pd.to_datetime(history["sale_date"])
    history = history.sort_values("sale_date")

    if len(history) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Need at least {MIN_HISTORY_DAYS} days of history before {target_date}, "
            f"got {len(history)}"
        )

    last_row = history.iloc[-1]
    target_date = pd.Timestamp(target_date)
    target_row = {
        "store_id": last_row["store_id"],
        "sale_date": target_date,
        "day_of_week": target_date.isoweekday(),
        "sales": 0.0,  # placeholder - never used by lag/rolling features
        "is_promo": is_promo,
        "is_school_holiday": is_school_holiday,
        "state_holiday": state_holiday,
        "store_type": last_row["store_type"],
        "assortment": last_row["assortment"],
        "competition_distance": last_row["competition_distance"],
        "has_promo2": last_row["has_promo2"],
    }
    combined = pd.concat([history, pd.DataFrame([target_row])], ignore_index=True)

    combined = add_calendar_features(combined)
    combined = _add_lag_and_rolling_features(combined)
    combined = _encode_categoricals(combined)

    return combined.iloc[[-1]][FEATURE_COLUMNS]
