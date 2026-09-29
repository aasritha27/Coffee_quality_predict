"""Write a deployment report for the current MLflow champion alias."""
import json
import sys

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from common import ARTIFACTS
from registry_utils import MODEL_NAME, configure_tracking


def generate_registry_report() -> bool:
    configure_tracking()
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        print("[ERROR] No champion alias found. Run the Lab 6 registry pipeline first.")
        return False

    run = client.get_run(champion.run_id)
    report = {
        "registry_status": "READY_FOR_DEPLOYMENT",
        "model_lineage": {
            "registered_name": MODEL_NAME,
            "version": int(champion.version),
            "alias": "champion",
            "run_id": champion.run_id,
            "model_uri": f"models:/{MODEL_NAME}@champion",
            "artifact_uri": champion.source,
        },
        "performance_metrics": run.data.metrics,
        "hyperparameters": run.data.params,
        "preprocessing_artifacts": [
            "preprocessing/imputer.pkl",
            "preprocessing/scaler.pkl",
            "preprocessing/ohe.pkl",
            "preprocessing/dataset_metadata.json",
        ],
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS / "production_model_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Champion version {champion.version} report saved to {report_path}.")
    return True


if __name__ == "__main__":
    sys.exit(0 if generate_registry_report() else 1)
