"""Shared MLflow tracking and registry settings for Labs 5-6."""
import mlflow
from common import ROOT

TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "Coffee_Quality_Production"
MODEL_NAME = "Coffee_Quality_Production_Model"


def configure_tracking() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=(ROOT / "mlruns").as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
