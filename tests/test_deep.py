import numpy as np
import pandas as pd
import torch

from demandcast.models.deep import (
    SEQ_LEN,
    LSTMForecaster,
    SlidingWindowDataset,
    build_store_index,
    fit_store_scalers,
    predict_lstm,
    train_lstm,
)


def _synthetic_panel(n_days: int = 90, store_ids=(1, 2, 3)) -> pd.DataFrame:
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
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_store_index_covers_every_store():
    df = _synthetic_panel()
    index = build_store_index(df)
    assert index.num_stores == 3
    assert set(index.id_to_idx) == {1, 2, 3}


def test_store_scalers_are_fit_per_store_from_train_only():
    df = _synthetic_panel()
    train_df = df[df["sale_date"] < "2015-03-01"]
    scalers = fit_store_scalers(train_df)

    assert set(scalers) == {1, 2, 3}
    # store 3 has a higher baseline than store 1, so its log-mean should be higher too
    assert scalers[3][0] > scalers[1][0]


def test_sliding_window_dataset_respects_seq_len_and_date_filter():
    df = _synthetic_panel(n_days=90)
    scalers = fit_store_scalers(df[df["sale_date"] < "2015-03-01"])
    index = build_store_index(df)

    dataset = SlidingWindowDataset(
        df,
        index,
        scalers,
        min_date=pd.Timestamp("2015-03-01"),
        max_date=pd.Timestamp("2015-03-31"),
    )

    assert len(dataset) == 31 * 3  # 31 target days x 3 stores
    seq, calendar, store_idx, target = dataset[0]
    assert seq.shape == (SEQ_LEN,)
    assert calendar.shape == (5,)
    assert isinstance(store_idx.item(), int)
    assert torch.isfinite(target)


def test_lstm_forward_pass_shape():
    model = LSTMForecaster(num_stores=3)
    seq = torch.zeros(4, SEQ_LEN)
    calendar = torch.zeros(4, 5)
    store_idx = torch.tensor([0, 1, 2, 0])
    output = model(seq, calendar, store_idx)
    assert output.shape == (4,)


def test_train_and_predict_lstm_beats_naive_mean():
    df = _synthetic_panel(n_days=90)
    index = build_store_index(df)
    train_df = df[df["sale_date"] < "2015-03-01"]
    test_df = df[df["sale_date"] >= "2015-03-01"]
    scalers = fit_store_scalers(train_df)

    train_dataset = SlidingWindowDataset(
        df, index, scalers, train_df["sale_date"].min(), pd.Timestamp("2015-02-14")
    )
    val_dataset = SlidingWindowDataset(
        df, index, scalers, pd.Timestamp("2015-02-15"), pd.Timestamp("2015-02-28")
    )
    test_dataset = SlidingWindowDataset(
        df, index, scalers, test_df["sale_date"].min(), test_df["sale_date"].max()
    )

    torch.manual_seed(0)
    model, history = train_lstm(
        train_dataset, val_dataset, index.num_stores, epochs=10, batch_size=32
    )
    preds = predict_lstm(model, test_dataset)

    naive_preds = np.full(len(test_dataset), train_df["sales"].mean())
    naive_rmse = np.sqrt(np.mean((np.array(test_dataset.targets_actual) - naive_preds) ** 2))
    model_rmse = np.sqrt(np.mean((np.array(test_dataset.targets_actual) - preds) ** 2))

    assert len(history.val_loss) > 0
    assert model_rmse < naive_rmse
