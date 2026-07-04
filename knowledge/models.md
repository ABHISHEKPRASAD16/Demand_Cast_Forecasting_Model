# Models in DemandCast

DemandCast compares two kinds of models: classical/statistical models fit on a single representative store, and global machine-learning models trained once across all 1,115 stores.

## Why store 262 is the representative store

Classical models (baselines, SARIMAX, Prophet) fit one model per time series. Fitting 1,115 of them would multiply runtime without teaching anything new about the method, so DemandCast picks one representative store instead: store 262, chosen because it has the highest total sales among stores with zero gaps in its open-day history (942 out of 942 possible open days present). This gives the classical models a clean, complete daily series to work with.

## Baseline models

- **Naive**: repeats the last observed sales value for every day of the forecast horizon.
- **Seasonal naive (7-day)**: repeats the last 7 days of sales cyclically across the horizon, capturing weekly seasonality (weekday effects, weekend closures) with no fitting at all.
- **Moving average (7-day)**: forecasts a flat line at the mean of the last 7 observed days.

Any fancier model has to beat seasonal-naive to be worth using — for daily retail sales, a 7-day lag is a genuinely strong baseline because weekly seasonality explains most of the variance on its own.

## SARIMAX

A seasonal ARIMA model. The seasonal component is fixed at (1,1,1,7) — one seasonal AR term, one seasonal difference, one seasonal MA term, at a 7-day period — based on known weekly seasonality in retail sales, rather than grid-searched (which would be slow). The non-seasonal (p, d, q) order is grid-searched by AIC over a small range, which is the standard, cheaper stand-in for an automatic order-search tool.

## Prophet

Facebook/Meta's Prophet model, configured with weekly and yearly seasonality and `is_promo` added as a known-in-advance regressor (a retailer knows its own promo calendar ahead of time, so this is a legitimate input rather than a leak).

## LightGBM (global model)

A single LightGBM gradient-boosted tree model trained across all 1,115 stores at once, using engineered features: lag features (sales 7/14/28 days ago), rolling statistics (7- and 28-day rolling mean and standard deviation, computed only from prior days so the current day's own sales never leaks into its own features), and calendar features (day of week, month, week of year, weekend flag, promo flag, school holiday flag, state holiday code), plus static store attributes (store type, assortment, competition distance, promo2 flag).

A single global model generalizes better than 1,115 separate per-store models: low-volume stores borrow statistical strength from similar ones, and it is what you would actually deploy, since retraining and serving 1,115 models is an operational burden a single global model avoids entirely. This is the model registered to the MLflow Model Registry and served by the API, because it is fast to retrain and, per the comparison results, within noise of the LSTM's global accuracy.

## LSTM (global model, PyTorch)

A recurrent neural network trained across all stores at once, with a per-store embedding layer capturing store-specific baseline effects. Unlike LightGBM, the LSTM is handed raw (normalized) sales sequences instead of lag/rolling features — learning temporal structure from the sequence is the whole point of a recurrent model, so re-deriving it by hand with engineered features would defeat the comparison. Sales values are normalized per-store (log1p, then z-scored using only that store's training-period mean/std) before being fed to the model, since sales scale varies by roughly 100x across stores.

## Feature engineering notes

`customers` and `is_open` are deliberately excluded from the LightGBM/LSTM feature set even though they exist in the underlying sales data: `customers` is itself an outcome correlated with sales that would not be known in advance of making a forecast, and `is_open` is constant `True` in the modeling data because closed days (where sales are trivially zero) are filtered out before feature engineering.
