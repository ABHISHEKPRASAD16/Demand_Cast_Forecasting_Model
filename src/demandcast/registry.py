"""Shared MLflow Model Registry access - used by both the drift monitor
(scores reference/current data with the deployed model, doesn't retrain)
and the serving API / dashboard (predicts with the deployed model, doesn't
retrain).
"""

import logging
import os
from pathlib import Path

import lightgbm as lgb
import mlflow
from mlflow.entities.model_registry import ModelVersion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "demandcast-lightgbm"

STATIC_MODEL_PATH_ENV = "DEMANDCAST_MODEL_PATH"
STATIC_MODEL_VERSION = "frozen-snapshot"


def get_latest_model_version(
    tracking_uri: str = MLFLOW_TRACKING_URI, model_name: str = MODEL_NAME
) -> ModelVersion:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions of '{model_name}' found - run "
            "`python -m demandcast.training.train --config config/models/lightgbm.yaml` first."
        )
    return max(versions, key=lambda v: int(v.version))


def load_latest_model(tracking_uri: str = MLFLOW_TRACKING_URI, model_name: str = MODEL_NAME):
    latest = get_latest_model_version(tracking_uri, model_name)
    logger.info("Loading %s version %s from the registry", model_name, latest.version)
    return mlflow.lightgbm.load_model(f"models:/{model_name}/{latest.version}")


def load_production_model(
    tracking_uri: str = MLFLOW_TRACKING_URI, model_name: str = MODEL_NAME
) -> tuple[lgb.Booster, str]:
    """Load the model to serve/predict with, plus a human-readable version label.

    If DEMANDCAST_MODEL_PATH is set, loads a plain LightGBM text file instead
    of going through the MLflow registry - used for the hosted Streamlit demo,
    which has no MLflow tracking store available and would otherwise hit the
    same local-artifact-path limitation documented in the README's Docker
    section. drift.py deliberately does NOT use this function: monitoring
    should always compare against whatever is actually in the registry.
    """
    static_path = os.environ.get(STATIC_MODEL_PATH_ENV)
    if static_path:
        logger.info("Loading static model export from %s", static_path)
        return lgb.Booster(model_file=static_path), STATIC_MODEL_VERSION

    version_info = get_latest_model_version(tracking_uri, model_name)
    logger.info("Loading %s version %s from the registry", model_name, version_info.version)
    model = mlflow.lightgbm.load_model(f"models:/{model_name}/{version_info.version}")
    return model, str(version_info.version)


def export_model_to_file(path: Path) -> None:
    """One-time step producing the static export DEMANDCAST_MODEL_PATH points at."""
    model = load_latest_model()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Exported model to %s", path)


def main() -> None:
    export_model_to_file(Path("models/lightgbm_export.txt"))


if __name__ == "__main__":
    main()
