from pathlib import Path

import pytest

from demandcast.training.config import (
    LightGBMTrainingConfig,
    LSTMTrainingConfig,
    load_training_config,
)


def _write(tmp_path: Path, text: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text)
    return config_path


def test_load_lightgbm_config_applies_defaults(tmp_path: Path):
    config_path = _write(tmp_path, "model_type: lightgbm\nparams:\n  learning_rate: 0.1\n")
    cfg = load_training_config(config_path)

    assert isinstance(cfg, LightGBMTrainingConfig)
    assert cfg.params == {"learning_rate": 0.1}
    assert cfg.test_weeks == 6
    assert cfg.register_model is False


def test_load_lstm_config_applies_defaults(tmp_path: Path):
    config_path = _write(tmp_path, "model_type: lstm\nepochs: 3\n")
    cfg = load_training_config(config_path)

    assert isinstance(cfg, LSTMTrainingConfig)
    assert cfg.epochs == 3
    assert cfg.hidden_size == 64


def test_unknown_model_type_raises(tmp_path: Path):
    config_path = _write(tmp_path, "model_type: xgboost\n")
    with pytest.raises(ValueError, match="Unknown model_type"):
        load_training_config(config_path)


def test_real_lightgbm_config_file_loads():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_training_config(repo_root / "config" / "models" / "lightgbm.yaml")
    assert isinstance(cfg, LightGBMTrainingConfig)
    assert cfg.register_model is True


def test_real_lstm_config_file_loads():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_training_config(repo_root / "config" / "models" / "lstm.yaml")
    assert isinstance(cfg, LSTMTrainingConfig)
