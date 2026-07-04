"""Pydantic schema for config/models/*.yaml - the whole point of a
"config-driven, not notebook-driven" training pipeline: every
hyperparameter a run used lives in a versioned YAML file (and gets logged
to MLflow as a param), not typed once into a notebook cell and forgotten.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class LightGBMTrainingConfig(BaseModel):
    model_type: Literal["lightgbm"]
    test_weeks: int = 6
    cv_val_weeks: int = 6
    num_boost_round: int = 1000
    early_stopping_rounds: int = 50
    params: dict = Field(default_factory=dict)
    register_model: bool = False


class LSTMTrainingConfig(BaseModel):
    model_type: Literal["lstm"]
    test_weeks: int = 6
    cv_val_weeks: int = 6
    epochs: int = 15
    batch_size: int = 512
    lr: float = 1e-3
    patience: int = 3
    embedding_dim: int = 8
    hidden_size: int = 64
    register_model: bool = False


TrainingConfig = LightGBMTrainingConfig | LSTMTrainingConfig


def load_training_config(path: Path) -> TrainingConfig:
    raw = yaml.safe_load(path.read_text())
    model_type = raw.get("model_type")
    if model_type == "lightgbm":
        return LightGBMTrainingConfig.model_validate(raw)
    if model_type == "lstm":
        return LSTMTrainingConfig.model_validate(raw)
    raise ValueError(f"Unknown model_type in {path}: {model_type!r}")
