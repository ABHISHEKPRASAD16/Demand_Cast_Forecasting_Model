"""A global LSTM: one recurrent model, all stores, a per-store embedding for
store-specific baseline effects.

Unlike LightGBM, this model is handed raw (normalized) sales sequences
instead of lag/rolling features - learning temporal structure from the
sequence is the LSTM's whole job, so re-deriving it by hand would defeat
the point of comparing the two approaches.

Sequences are built over each store's chronological run of *open* days -
the rows fct_sales already restricts to - not literal calendar days, so a
handful of closed-day gaps get compressed rather than represented. That's
a real simplification worth naming, not one to gloss over.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from demandcast.features.engineering import add_calendar_features

SEQ_LEN = 28
CALENDAR_FEATURES = ("day_of_week", "is_weekend", "is_promo", "is_school_holiday", "month")


@dataclass
class StoreIndex:
    """Maps store_id -> a contiguous 0..N-1 index for nn.Embedding."""

    id_to_idx: dict[int, int]

    @property
    def num_stores(self) -> int:
        return len(self.id_to_idx)


def build_store_index(df: pd.DataFrame) -> StoreIndex:
    unique_ids = sorted(df["store_id"].unique())
    return StoreIndex({store_id: idx for idx, store_id in enumerate(unique_ids)})


def fit_store_scalers(train_df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Per-store (mean, std) of log1p(sales), fit on the training split only.

    Sales scale varies by ~100x across stores; normalizing per-store (rather
    than globally) is what lets the LSTM share weights across stores without
    high-volume stores dominating the loss.
    """
    log_sales = np.log1p(train_df["sales"])
    grouped = log_sales.groupby(train_df["store_id"])
    means = grouped.mean()
    stds = grouped.std().replace(0, np.nan).fillna(1.0)
    return {store_id: (float(means[store_id]), float(stds[store_id])) for store_id in means.index}


class SlidingWindowDataset(Dataset):
    """One sample per (store, target day): the SEQ_LEN normalized sales values
    immediately before it plus its own calendar features.

    `min_date`/`max_date` filter which days are used as *targets*, but the
    input window for a target near a split boundary still draws its history
    from whatever came before it chronologically - using past data to predict
    the future isn't leakage, it's the entire point.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        store_index: StoreIndex,
        store_scalers: dict[int, tuple[float, float]],
        min_date: pd.Timestamp,
        max_date: pd.Timestamp,
        seq_len: int = SEQ_LEN,
    ):
        df = add_calendar_features(df.copy())
        df = df.sort_values(["store_id", "sale_date"]).reset_index(drop=True)

        self.store_scalers = store_scalers
        self.sequences: list[np.ndarray] = []
        self.calendar_feats: list[np.ndarray] = []
        self.store_idxs: list[int] = []
        self.store_ids: list[int] = []
        self.targets_normalized: list[float] = []
        self.targets_actual: list[float] = []
        self.target_dates: list[pd.Timestamp] = []

        for store_id, group in df.groupby("store_id", sort=False):
            if store_id not in store_scalers:
                continue
            mean, std = store_scalers[store_id]
            raw_sales = group["sales"].to_numpy(dtype=float)
            normalized = (np.log1p(raw_sales) - mean) / std
            calendar = group[list(CALENDAR_FEATURES)].to_numpy(dtype=float)
            dates = group["sale_date"].to_numpy()
            store_idx = store_index.id_to_idx[store_id]

            for i in range(seq_len, len(group)):
                target_date = dates[i]
                if not (min_date <= target_date <= max_date):
                    continue
                self.sequences.append(normalized[i - seq_len : i])
                self.calendar_feats.append(calendar[i])
                self.store_idxs.append(store_idx)
                self.store_ids.append(store_id)
                self.targets_normalized.append(normalized[i])
                self.targets_actual.append(raw_sales[i])
                self.target_dates.append(target_date)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.calendar_feats[idx], dtype=torch.float32),
            torch.tensor(self.store_idxs[idx], dtype=torch.long),
            torch.tensor(self.targets_normalized[idx], dtype=torch.float32),
        )


class LSTMForecaster(nn.Module):
    def __init__(self, num_stores: int, embedding_dim: int = 8, hidden_size: int = 64):
        super().__init__()
        self.store_embedding = nn.Embedding(num_stores, embedding_dim)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim + len(CALENDAR_FEATURES), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, seq: torch.Tensor, calendar: torch.Tensor, store_idx: torch.Tensor
    ) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq.unsqueeze(-1))
        hidden = h_n[-1]
        embedded = self.store_embedding(store_idx)
        combined = torch.cat([hidden, embedded, calendar], dim=1)
        return self.head(combined).squeeze(-1)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def train_lstm(
    train_dataset: SlidingWindowDataset,
    val_dataset: SlidingWindowDataset,
    num_stores: int,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 3,
) -> tuple[LSTMForecaster, TrainingHistory]:
    """Train with early stopping on val_loss - the same principle as LightGBM's
    early stopping, just implemented by hand since PyTorch has no built-in."""
    model = LSTMForecaster(num_stores)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = TrainingHistory()
    best_val_loss, best_state, epochs_without_improvement = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        train_losses = []
        for seq, calendar, store_idx, target in train_loader:
            optimizer.zero_grad()
            pred = model(seq, calendar, store_idx)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for seq, calendar, store_idx, target in val_loader:
                pred = model(seq, calendar, store_idx)
                val_losses.append(loss_fn(pred, target).item())

        history.train_loss.append(float(np.mean(train_losses)))
        history.val_loss.append(float(np.mean(val_losses)))

        if history.val_loss[-1] < best_val_loss:
            best_val_loss = history.val_loss[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


def predict_lstm(model: LSTMForecaster, dataset: SlidingWindowDataset) -> np.ndarray:
    """Predictions in the original sales scale, in dataset order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    normalized_preds: list[float] = []
    with torch.no_grad():
        for seq, calendar, store_idx, _ in loader:
            normalized_preds.extend(model(seq, calendar, store_idx).tolist())

    means = np.array([dataset.store_scalers[sid][0] for sid in dataset.store_ids])
    stds = np.array([dataset.store_scalers[sid][1] for sid in dataset.store_ids])
    log_sales = np.asarray(normalized_preds) * stds + means
    return np.expm1(log_sales)
