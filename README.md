# DemandCast

Production-grade demand forecasting and AI insights platform. Ingests retail sales data, forecasts demand with classical and deep learning models, tracks and versions every experiment, serves predictions via an API, monitors for drift, and lets business users ask questions in natural language through a RAG-powered AI analyst agent.

Dataset: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales) (Kaggle) — daily sales for 1,115 stores.

## Architecture

_Diagram to be added once Phase 4 (serving) exists._

```
Kaggle CSVs → data/raw → dbt (staging → marts) → DuckDB warehouse → [models] → [API] → [dashboard/agent]
```

## Roadmap

- [x] **Phase 1 — Data pipeline**: raw/staging/marts layering with dbt + DuckDB, source-level tests, documented lineage.
- [x] **Phase 2 — Modeling**: naive/seasonal-naive baselines → ARIMA/Prophet → global LightGBM → global LSTM, with a MAPE/RMSE/WAPE comparison table.
- [x] **Phase 3 — MLOps**: MLflow tracking + registry, config-driven training, Great Expectations input validation, Evidently drift monitoring, pytest CI + build.
- [x] **Phase 4 — Serving**: FastAPI predict/metadata/health endpoints, Docker + docker-compose.
- [ ] **Phase 5 — AI Analyst**: Streamlit scenario dashboard, LangChain RAG agent over model docs/metrics/forecasts with citations and guardrails.

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\Activate.ps1 in PowerShell
pip install -r requirements-dev.txt -e .
pre-commit install
```

### 1. Get a Kaggle API token

Kaggle → Account → *Create New Token*, save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`.

### 2. Pull the raw data

```bash
python -m demandcast.ingestion.download_rossmann
```

Downloads `train.csv` and `store.csv` into `data/raw/`.

### 3. Build the warehouse

```bash
cd dbt/demandcast
cp profiles.yml.example ~/.dbt/profiles.yml   # or set DBT_PROFILES_DIR to this folder
dbt deps
dbt run
dbt test
```

This builds `staging.stg_train`, `staging.stg_store`, and `marts.fct_sales` in `data/warehouse.duckdb`, and runs not-null/unique/referential-integrity tests on each.

### 4. Run the test suite

```bash
pytest
```

## Phase 2 results

Every model is evaluated against the same held-out test window: the final 6 weeks of calendar time (~Aug 2015, matching Rossmann's actual Kaggle competition test period). WAPE is the primary metric — unlike MAPE it's defined even on zero-sales days and weights by sales volume rather than by row count, so one low-volume store-day can't dominate the score (see [metrics.py](src/demandcast/evaluation/metrics.py) for the full reasoning).

**Same-store comparison** — store 262 (highest total sales, zero gaps in its open-day history), so every model family is judged on identical ground:

| Model | MAPE | RMSE | WAPE |
|---|---|---|---|
| Naive (last value) | 15.01 | 4508.36 | 15.73 |
| Moving average (7d) | 15.65 | 4489.00 | 15.98 |
| Seasonal naive (7d) | 9.06 | 2110.39 | 8.64 |
| SARIMAX(2,1,1)(1,1,1,7) | 9.21 | 2223.46 | 8.72 |
| Prophet | 7.83 | 2046.24 | 8.11 |
| LightGBM (global model, this store's rows) | 5.85 | 1660.39 | 6.12 |
| LSTM (global model, this store's rows) | 6.27 | 1993.25 | 6.62 |

**Global models** — LightGBM and the LSTM are each trained once across all 1,115 stores (the realistic production setup, rather than 1,115 separate models):

| Model | Scope | MAPE | RMSE | WAPE |
|---|---|---|---|---|
| LightGBM | All 1,115 stores | 9.02 | 902.21 | 8.75 |
| LSTM | All 1,115 stores | 8.63 | 901.50 | 8.59 |

Takeaways:
- Weekly seasonality dominates: seasonal-naive alone nearly matches SARIMAX, and beats the naive/moving-average baselines by ~40% WAPE.
- Prophet's built-in yearly+weekly seasonality plus a promo regressor edges out a manually order-selected SARIMAX.
- The global LightGBM and LSTM — trained once across every store, not tuned per series — still beat every classical model *on that same store*, because they borrow statistical strength from the other 1,114 series. That's the actual case for a global model in production: fewer models to retrain and serve, and better accuracy than a one-off per store.
- LightGBM edges out the LSTM on the store-262 subset; the LSTM edges out LightGBM globally. Consistent with expectations — engineered lag/rolling features are a strong, low-variance signal for a single well-behaved series, while the LSTM's learned representations generalize slightly better in aggregate across a very heterogeneous panel of stores.

Reproduce with:

```bash
python -m demandcast.evaluation.run_comparison
```

This writes [reports/model_comparison.csv](reports/model_comparison.csv) and `reports/model_comparison.md`. LSTM training is the slow part (~15-20 min on CPU); everything else finishes in well under a minute.

## Phase 3 — MLOps

Three new tools, wired around the models from Phase 2:

**Great Expectations** validates `fct_sales` right after it's loaded from the warehouse, before any feature engineering — a different check than Phase 1's dbt tests. Those validate the warehouse is well-formed *at build time*; this validates that the exact DataFrame a training run is about to consume still looks sane *at run time* (nulls, sales ≥ 0, day_of_week in 1–7, known store types, etc.), without assuming dbt was recently and correctly rerun. See [expectations.py](src/demandcast/validation/expectations.py).

**Config-driven training + MLflow** replaces the hardcoded hyperparameters from Phase 2 with versioned YAML ([config/models/lightgbm.yaml](config/models/lightgbm.yaml), [config/models/lstm.yaml](config/models/lstm.yaml)), validated by pydantic. Every run logs its params, metrics, and model artifact to MLflow (SQLite backend — the plain file store is in maintenance mode as of MLflow 3.x), so runs are versioned and comparable instead of numbers only living in a printed table:

```bash
python -m demandcast.training.train --config config/models/lightgbm.yaml
python -m demandcast.training.train --config config/models/lstm.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs at localhost:5000
```

LightGBM is registered to the Model Registry (`demandcast-lightgbm`) — it's fast to retrain and, per the Phase 2 results, within noise of the LSTM globally, which makes it the more sensible thing to actually serve.

**Evidently drift monitoring** answers a specific question: would a monitoring job have caught a distribution shift between what the model trained on and the most recent data it's asked to predict on? [drift.py](src/demandcast/monitoring/drift.py) pulls the *latest registered* LightGBM version from the Model Registry (it doesn't retrain), scores the training window (reference) and test window (current), and runs Evidently's `DataDriftPreset` + `RegressionPreset` to check both feature drift and prediction/target drift:

```bash
python -m demandcast.monitoring.drift   # writes reports/drift/drift_report.html
```

The report is self-contained HTML (~4MB) and gitignored rather than versioned like Phase 2's comparison table — it's regenerated on demand, not a number worth diffing in git history.

## Phase 4 — Serving

```bash
uvicorn demandcast.serving.main:app --reload   # or: make serve
```

- `GET /health` — liveness check.
- `GET /model/metadata` — the currently registered LightGBM version, its MLflow run id, and its training-time test metrics.
- `POST /predict` — forecast sales for one `(store_id, date)`:

  ```bash
  curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
    -d '{"store_id": 262, "date": "2015-08-01", "is_promo": true}'
  ```

**Why a bare `(store_id, date)` isn't enough on its own:** the model was trained on lag/rolling features (last-7/14/28-day sales, rolling mean/std), so predicting requires that store's recent history, not just the request payload. `/predict` pulls the store's trailing ~45 days from the DuckDB warehouse and runs them through [`build_features_for_prediction`](src/demandcast/features/engineering.py), the same lag/rolling/calendar logic `build_features` uses at training time, just applied to one store's window instead of the whole panel — verified with a test asserting the two paths produce identical features for the same row. Only `is_promo`/`is_school_holiday`/`state_holiday` come from the request itself, since those are calendar/business facts the caller would know in advance, not something in the historical data.

The model, its version, and the history-fetching logic are all `Depends()`-injected, so [tests/test_serving.py](tests/test_serving.py) overrides them with stubs — no real warehouse or MLflow registry needed to run the test suite (CI has neither).

### Docker

```bash
make docker-build
make docker-up      # API at localhost:8000
```

**Known limitation:** local MLflow's file-based artifact store bakes the *absolute host path* into each run (`file:///…/mlruns/…`), not a relative or container-portable one. A model registered by running training directly on the host will fail to load inside the container, since that absolute path doesn't exist there. This isn't a bug in this repo's code — it's inherent to a local, non-server MLflow setup; the real fix is a proper MLflow Tracking Server backed by an object store (S3/Azure Blob) with location-independent artifact URIs, which is the natural next step before any real deployment. Docker itself wasn't available in the environment this was built in, so the image/compose file are written to the same standards as everything else here but not build-verified end-to-end — flagging that plainly rather than claiming otherwise.

## Project layout

```
src/demandcast/
  config/           # pydantic schema + loader for config/pipeline.yaml
  ingestion/        # Kaggle download script
  data/             # DuckDB -> pandas loader for the marts layer
  features/         # lag/rolling/calendar feature engineering (LightGBM)
  models/           # baselines, SARIMAX/Prophet, LightGBM, LSTM (PyTorch)
  evaluation/       # MAPE/RMSE/WAPE, time-based CV splits, the comparison-table runner
  validation/       # Great Expectations gate for fct_sales, pre-training
  training/         # config-driven training entrypoint + MLflow logging
  monitoring/       # Evidently drift report against the registered model
  serving/          # FastAPI app: predict / model metadata / health
  registry.py       # shared MLflow Model Registry loader (drift + serving)
dbt/demandcast/     # dbt project: staging + marts models, source/model tests
config/             # YAML config, validated via pydantic (src/demandcast/config, training/config)
data/               # raw/staging/marts + warehouse.duckdb (gitignored, pulled/generated)
notebooks/          # EDA only — no pipeline logic lives here
reports/            # model_comparison.csv/.md (versioned) + drift/*.html (gitignored, regenerated)
tests/              # pytest unit/integration tests
mlflow.db           # local MLflow tracking store (gitignored, created on first training run)
Dockerfile, docker-compose.yml, requirements-serving.txt   # the API's slim runtime image
```

## Why this project

Built to take a real production forecasting problem (previously improved a demand/volume forecast at a logistics company by ~8%) through the full lifecycle an industry ML team would expect: tested data pipeline, experiment tracking and model registry, drift monitoring, a served API, and an LLM layer grounded in the platform's own metrics and forecasts.

## License

MIT — see [LICENSE](LICENSE).
