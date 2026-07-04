import lightgbm as lgb
import numpy as np

from demandcast.registry import STATIC_MODEL_VERSION, load_production_model


def test_load_production_model_uses_static_path_when_env_var_set(tmp_path, monkeypatch):
    X = np.random.default_rng(0).random((20, 2))
    y = np.random.default_rng(0).random(20)
    booster = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(X, label=y))

    model_path = tmp_path / "model.txt"
    booster.save_model(str(model_path))
    monkeypatch.setenv("DEMANDCAST_MODEL_PATH", str(model_path))

    model, version = load_production_model()

    assert version == STATIC_MODEL_VERSION
    assert model.predict(X).shape == (20,)
