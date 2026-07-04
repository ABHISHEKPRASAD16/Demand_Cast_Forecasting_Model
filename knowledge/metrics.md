# Forecast accuracy metrics

DemandCast reports three metrics for every model: MAPE, RMSE, and WAPE. WAPE is the primary metric used to compare models and pick a winner; MAPE and RMSE are reported for familiarity and scale context.

## WAPE (Weighted Absolute Percentage Error) — primary metric

WAPE = sum(|actual - predicted|) / sum(|actual|), expressed as a percentage.

It is computed once over the whole set of rows being evaluated (summing errors and summing actuals separately, then dividing), not averaged row-by-row. This means it is defined even when some actual values are zero, and it naturally weights high-volume days/stores more heavily than low-volume ones — which matches how a retail business actually cares about forecast error: getting a big store's forecast wrong matters more than getting a tiny store's forecast wrong by the same percentage.

## MAPE (Mean Absolute Percentage Error)

MAPE = mean(|actual - predicted| / |actual|), expressed as a percentage, averaged per row.

MAPE is undefined when an actual value is exactly zero (division by zero), and DemandCast's implementation drops those rows rather than crashing. MAPE also treats every row equally regardless of its sales volume, so a handful of very low-volume store-days can swing the metric even though they're commercially unimportant. It is reported because stakeholders usually ask for it by name, but it is not what drives model selection in this project.

## RMSE (Root Mean Squared Error)

RMSE = sqrt(mean((actual - predicted)^2)), in the same units as sales (not a percentage).

RMSE penalizes large individual errors more heavily than small ones (because errors are squared before averaging), which makes it useful for understanding whether a model has occasional big misses even if its average percentage error looks fine.

## Why the holdout window is 6 weeks

Every model in DemandCast is evaluated against the same held-out test window: the final 6 weeks of calendar time in the dataset (approximately August 2015). This mirrors the actual test period used in the real Rossmann Kaggle competition, and using one shared cutoff date (rather than a per-store percentage split) means every store is evaluated on the same forecast horizon — which matches how the model would actually be used in production, where you forecast the same upcoming period for every store at once.
