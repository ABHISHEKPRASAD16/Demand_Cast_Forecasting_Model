.PHONY: install install-dev format lint test ingest dbt-run dbt-test dbt-docs clean

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

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
