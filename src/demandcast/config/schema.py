"""Pydantic schema for config/pipeline.yaml, plus the YAML loader.

Validating config at load time (rather than trusting raw dict access
scattered through the codebase) turns a typo'd YAML key into a startup
error instead of a silent None deep in a pipeline run.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# Every entrypoint in this project (Makefile targets, README commands, the
# Docker image's WORKDIR) is run with the repo root as cwd, so that's the
# right default - unlike `__file__`-relative traversal, it still resolves
# correctly once the package is pip-installed non-editably (e.g. in Docker),
# where the installed file's parents no longer line up with the repo layout.
PROJECT_ROOT = Path(os.environ.get("DEMANDCAST_PROJECT_ROOT", Path.cwd()))
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


class DatasetConfig(BaseModel):
    kaggle_slug: str
    raw_dir: Path
    required_files: list[str] = Field(default_factory=list)


class WarehouseConfig(BaseModel):
    duckdb_path: Path
    staging_schema: str
    marts_schema: str


class ModelingConfig(BaseModel):
    representative_store_id: int
    test_weeks: int = 6
    cv_val_weeks: int = 6


class PipelineConfig(BaseModel):
    dataset: DatasetConfig
    warehouse: WarehouseConfig
    modeling: ModelingConfig


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load and validate pipeline.yaml. Paths are resolved relative to the repo root."""
    config_path = path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text())
    config = PipelineConfig.model_validate(raw)

    config.dataset.raw_dir = PROJECT_ROOT / config.dataset.raw_dir
    config.warehouse.duckdb_path = PROJECT_ROOT / config.warehouse.duckdb_path
    return config
