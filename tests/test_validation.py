import pandas as pd
import pytest

from demandcast.validation import DataValidationError, validate_fct_sales_or_raise


def _valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": [1, 2, 3],
            "sale_date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"]),
            "day_of_week": [4, 5, 6],
            "sales": [100, 200, 300],
            "customers": [10, 20, 30],
            "is_open": [True, True, True],
            "is_promo": [False, True, False],
            "state_holiday": ["0", "0", "0"],
            "is_school_holiday": [False, False, False],
            "store_type": ["a", "b", "c"],
            "assortment": ["a", "a", "b"],
            "competition_distance": [500.0, None, 1200.0],
            "has_promo2": [False, True, False],
        }
    )


def test_valid_fct_sales_passes():
    validate_fct_sales_or_raise(_valid_df())  # should not raise


def test_negative_sales_fails():
    df = _valid_df()
    df.loc[0, "sales"] = -10
    with pytest.raises(DataValidationError):
        validate_fct_sales_or_raise(df)


def test_null_store_id_fails():
    df = _valid_df()
    df.loc[0, "store_id"] = None
    with pytest.raises(DataValidationError):
        validate_fct_sales_or_raise(df)


def test_closed_day_row_fails():
    df = _valid_df()
    df.loc[0, "is_open"] = False
    with pytest.raises(DataValidationError):
        validate_fct_sales_or_raise(df)


def test_unknown_store_type_fails():
    df = _valid_df()
    df.loc[0, "store_type"] = "z"
    with pytest.raises(DataValidationError):
        validate_fct_sales_or_raise(df)


def test_null_competition_distance_is_allowed():
    df = _valid_df()
    df["competition_distance"] = [None, None, None]
    validate_fct_sales_or_raise(df)  # should not raise
