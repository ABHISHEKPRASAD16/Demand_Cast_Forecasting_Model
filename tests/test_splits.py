import pandas as pd

from demandcast.evaluation.splits import expanding_window_splits, train_test_split_by_date


def _daily_df(n_days: int) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"sale_date": dates, "sales": range(n_days)})


def test_train_test_split_by_date_holds_out_exact_window():
    df = _daily_df(100)

    train, test = train_test_split_by_date(df, "sale_date", test_weeks=2)

    assert train["sale_date"].max() < test["sale_date"].min()
    assert len(test) == 14
    assert len(train) + len(test) == len(df)


def test_train_test_split_uses_max_date_in_data_not_today():
    df = _daily_df(30)
    _, test = train_test_split_by_date(df, "sale_date", test_weeks=1)
    assert test["sale_date"].max() == df["sale_date"].max()


def test_expanding_window_splits_are_chronological_and_non_overlapping():
    df = _daily_df(120)

    folds = list(expanding_window_splits(df, "sale_date", n_splits=3, val_weeks=2))

    assert len(folds) == 3
    for train_fold, val_fold in folds:
        assert train_fold["sale_date"].max() < val_fold["sale_date"].min()
        assert len(val_fold) == 14

    # each successive fold's training window should grow (expanding, not sliding)
    train_sizes = [len(train_fold) for train_fold, _ in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]


def test_expanding_window_splits_skip_folds_with_insufficient_history():
    df = _daily_df(10)
    folds = list(expanding_window_splits(df, "sale_date", n_splits=5, val_weeks=2))
    assert all(not train_fold.empty for train_fold, _ in folds)
