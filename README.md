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
- [ ] **Phase 2 — Modeling**: naive/seasonal-naive baselines → ARIMA/Prophet → LightGBM → LSTM, with a MAPE/RMSE/WAPE comparison table.
- [ ] **Phase 3 — MLOps**: MLflow tracking + registry, config-driven training, Great Expectations/dbt input validation, Evidently drift monitoring, pytest CI.
- [ ] **Phase 4 — Serving**: FastAPI predict/metadata/health endpoints, Docker + docker-compose.
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

## Project layout

```
src/demandcast/     # importable package — config, ingestion, (models/serving to come)
dbt/demandcast/     # dbt project: staging + marts models, source/model tests
config/             # YAML config, validated via pydantic (src/demandcast/config)
data/               # raw/staging/marts + warehouse.duckdb (gitignored, pulled/generated)
notebooks/          # EDA only — no pipeline logic lives here
tests/              # pytest unit/integration tests
```

## Why this project

Built to take a real production forecasting problem (previously improved a demand/volume forecast at a logistics company by ~8%) through the full lifecycle an industry ML team would expect: tested data pipeline, experiment tracking and model registry, drift monitoring, a served API, and an LLM layer grounded in the platform's own metrics and forecasts.

## License

MIT — see [LICENSE](LICENSE).
