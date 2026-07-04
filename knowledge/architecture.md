# DemandCast architecture

DemandCast is a demand forecasting platform for Rossmann Store Sales data (daily sales for 1,115 stores, January 2013 to July 2015), built in five phases.

## Data pipeline

Raw CSVs (`train.csv`, `store.csv`) are downloaded from Kaggle and loaded into a local DuckDB warehouse via dbt. dbt builds a staging layer (typed, renamed columns, one model per source table) and a marts layer (`fct_sales`, one row per store-date, joining sales with store attributes). dbt tests check not-null constraints, uniqueness of (store, date) combinations, and referential integrity between the fact table and the store dimension.

## Modeling

Models are compared against the same held-out test window (the final 6 weeks of the dataset). Classical models (naive, seasonal-naive, moving average, SARIMAX, Prophet) are fit on one representative store (store 262). LightGBM and an LSTM are trained once, globally, across all 1,115 stores.

## MLOps

Before training, a Great Expectations suite validates the sales data actually loaded for that run (not just the warehouse at build time) — checking things like no nulls in key columns, non-negative sales, and known store types. Training is config-driven: hyperparameters live in versioned YAML files, not hardcoded in scripts or typed once into a notebook. Every training run logs its parameters, metrics, and model artifact to MLflow; the LightGBM model is registered to the MLflow Model Registry as `demandcast-lightgbm`. A separate drift-monitoring job loads the latest registered model from the registry (without retraining) and compares the training window against the test window for both feature drift and prediction drift, using Evidently.

## Serving

A FastAPI service exposes `POST /predict` (forecast sales for a given store and date), `GET /model/metadata` (which model version is deployed and its training-time metrics), and `GET /health`. Because the model was trained on lag/rolling features, a prediction request needs that store's recent sales history, not just the store id and date in the request — the API fetches the store's trailing history from the DuckDB warehouse and recomputes the same lag/rolling/calendar features used at training time.

## AI Analyst layer

A Streamlit dashboard shows historical sales and forecasts per store, with a promo-toggle scenario control backed by the real model (since promo is an actual model input) plus an optional manual percentage-adjustment slider (explicitly labeled as a manual override, not a model-driven estimate). A retrieval-augmented chat agent, built with LangChain and Claude, can answer natural-language questions about the platform's own metrics, model design decisions, and per-store sales patterns, grounding its answers in retrieved documents and citing them, and can call the live forecast API as a tool to answer "what if" scenario questions.
