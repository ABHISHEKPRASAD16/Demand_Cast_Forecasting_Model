import pandas as pd
import pytest

from demandcast.features.engineering import (
    FEATURE_COLUMNS,
    LAG_DAYS,
    MIN_HISTORY_DAYS,
    ROLLING_WINDOWS,
    build_features,
    build_features_for_prediction,
)


def _synthetic_df(n_days: int = 40, store_ids=(1, 2)) -> pd.DataFrame:
    frames = []
    for store_id in store_ids:
        dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "store_id": store_id,
                    "sale_date": dates,
                    "day_of_week": dates.dayofweek + 1,
                    "sales": range(1, n_days + 1),
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


def test_build_features_drops_rows_without_full_lookback():
    df = _synthetic_df(n_days=40)
    result = build_features(df)

    max_lookback = max(max(LAG_DAYS), max(ROLLING_WINDOWS))
    per_store_rows = (40 - max_lookback) * 2  # two stores
    assert len(result.df) == per_store_rows
    assert result.df[[f"lag_{lag}" for lag in LAG_DAYS]].isna().sum().sum() == 0


def test_lag_features_reference_the_correct_prior_day():
    df = _synthetic_df(n_days=40, store_ids=(1,))
    result = build_features(df)

    row = result.df[result.df["sale_date"] == "2015-02-08"].iloc[0]
    expected_lag_7_date = row["sale_date"] - pd.Timedelta(days=7)
    expected_value = df.loc[df["sale_date"] == expected_lag_7_date, "sales"].item()
    assert row["lag_7"] == expected_value


def test_rolling_mean_excludes_current_day():
    df = _synthetic_df(n_days=40, store_ids=(1,))
    result = build_features(df)

    row = result.df.iloc[0]
    window_dates = pd.date_range(end=row["sale_date"] - pd.Timedelta(days=1), periods=7, freq="D")
    expected_mean = df.loc[df["sale_date"].isin(window_dates), "sales"].mean()
    assert row["rolling_mean_7"] == expected_mean


def test_categorical_columns_are_encoded_as_category_dtype():
    df = _synthetic_df()
    result = build_features(df)
    for col in result.categorical_columns:
        assert str(result.df[col].dtype) == "category"


def test_lag_features_do_not_leak_across_stores():
    df = _synthetic_df(n_days=40, store_ids=(1, 2))
    df.loc[df["store_id"] == 2, "sales"] = df.loc[df["store_id"] == 2, "sales"] * 100

    result = build_features(df)

    store_1_lags = result.df.loc[result.df["store_id"] == "1", "lag_7"]
    store_2_lags = result.df.loc[result.df["store_id"] == "2", "lag_7"]
    assert store_1_lags.max() < store_2_lags.min()


def test_build_features_for_prediction_matches_training_pipeline():
    df = _synthetic_df(n_days=40, store_ids=(1,))
    trained = build_features(df)

    target_date = pd.Timestamp("2015-02-08")
    target_row = trained.df[trained.df["sale_date"] == target_date].iloc[0]
    history = df[df["sale_date"] < target_date]

    served = build_features_for_prediction(
        history,
        target_date=target_date,
        is_promo=bool(target_row["is_promo"]),
        is_school_holiday=bool(target_row["is_school_holiday"]),
        state_holiday=str(target_row["state_holiday"]),
    )

    assert len(served) == 1
    for col in FEATURE_COLUMNS:
        assert served.iloc[0][col] == target_row[col], f"mismatch on {col}"


def test_build_features_for_prediction_raises_on_insufficient_history():
    df = _synthetic_df(n_days=10, store_ids=(1,))
    with pytest.raises(ValueError, match="at least"):
        build_features_for_prediction(
            df,
            target_date=pd.Timestamp("2015-01-11"),
            is_promo=False,
            is_school_holiday=False,
        )


def test_build_features_for_prediction_min_history_days_matches_lookback():
    assert max(max(LAG_DAYS), max(ROLLING_WINDOWS)) == MIN_HISTORY_DAYS
