"""Shared MLflow Model Registry access - used by both the drift monitor
(scores reference/current data with the deployed model, doesn't retrain)
and the serving API (predicts with the deployed model, doesn't retrain).
"""

import logging

import mlflow
from mlflow.entities.model_registry import ModelVersion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "demandcast-lightgbm"


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
