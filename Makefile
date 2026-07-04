.PHONY: install install-dev format lint test build ingest dbt-run dbt-test dbt-docs compare train-lightgbm train-lstm drift-report mlflow-ui clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

format:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	black --check src tests

test:
	pytest --cov=src/demandcast --cov-report=term-missing

build:
	python -m build

ingest:
	python -m demandcast.ingestion.download_rossmann

dbt-deps:
	cd dbt/demandcast && dbt deps

dbt-run:
	cd dbt/demandcast && dbt run

dbt-test:
	cd dbt/demandcast && dbt test

dbt-docs:
	cd dbt/demandcast && dbt docs generate && dbt docs serve

compare:
	python -m demandcast.evaluation.run_comparison

train-lightgbm:
	python -m demandcast.training.train --config config/models/lightgbm.yaml

train-lstm:
	python -m demandcast.training.train --config config/models/lstm.yaml

drift-report:
	python -m demandcast.monitoring.drift

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
