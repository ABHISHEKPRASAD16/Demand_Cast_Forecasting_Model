"""Pydantic schema for config/pipeline.yaml, plus the YAML loader.

Validating config at load time (rather than trusting raw dict access
scattered through the codebase) turns a typo'd YAML key into a startup
error instead of a silent None deep in a pipeline run.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


class DatasetConfig(BaseModel):
    kaggle_slug: str
    raw_dir: Path
    required_files: list[str] = Field(default_factory=list)


class WarehouseConfig(BaseModel):
    duckdb_path: Path
    staging_schema: str
    marts_schema: str


class PipelineConfig(BaseModel):
    dataset: DatasetConfig
    warehouse: WarehouseConfig


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load and validate pipeline.yaml. Paths are resolved relative to the repo root."""
    config_path = path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text())
    config = PipelineConfig.model_validate(raw)

    config.dataset.raw_dir = PROJECT_ROOT / config.dataset.raw_dir
    config.warehouse.duckdb_path = PROJECT_ROOT / config.warehouse.duckdb_path
    return config
