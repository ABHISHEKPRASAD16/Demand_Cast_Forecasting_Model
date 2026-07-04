from pathlib import Path

import pytest
from pydantic import ValidationError

from demandcast.config import load_config
from demandcast.config.schema import PROJECT_ROOT, PipelineConfig


def test_load_config_returns_validated_pipeline_config():
    config = load_config()

    assert isinstance(config, PipelineConfig)
    assert config.dataset.kaggle_slug == "pratyushakar/rossmann-store-sales"
    assert config.dataset.required_files == ["train.csv", "store.csv"]


def test_load_config_resolves_paths_relative_to_project_root():
    config = load_config()

    assert config.dataset.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert config.warehouse.duckdb_path == PROJECT_ROOT / "data" / "warehouse.duckdb"


def test_load_config_rejects_missing_required_field(tmp_path: Path):
    bad_config = tmp_path / "pipeline.yaml"
    bad_config.write_text("dataset:\n  kaggle_slug: 'x/y'\n")

    with pytest.raises(ValidationError):
        load_config(bad_config)
