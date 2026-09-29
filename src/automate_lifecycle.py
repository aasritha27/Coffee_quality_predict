"""Compare the candidate with the champion and update MLflow model aliases."""
import sys

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from registry_utils import MODEL_NAME, configure_tracking

METRIC_TO_MAXIMIZE = "r2"


def promote_candidate() -> bool:
    configure_tracking()
    client = MlflowClient()
    try:
        candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")
    except MlflowException:
        print("[ERROR] No candidate alias found. Run src/train_registry.py first.")
        return False

    candidate_run = client.get_run(candidate.run_id)
    candidate_score = candidate_run.data.metrics.get(METRIC_TO_MAXIMIZE)
    if candidate_score is None:
        print(f"[ERROR] Candidate version {candidate.version} has no {METRIC_TO_MAXIMIZE} metric.")
        return False

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        champion = None

    if champion is None:
        promote = True
        print("[INFO] No champion exists; candidate will be promoted.")
    elif champion.version == candidate.version:
        print(f"[INFO] Version {candidate.version} is already the champion.")
        client.delete_registered_model_alias(MODEL_NAME, "candidate")
        return True
    else:
        champion_run = client.get_run(champion.run_id)
        champion_score = champion_run.data.metrics.get(METRIC_TO_MAXIMIZE)
        if champion_score is None:
            print(f"[ERROR] Champion version {champion.version} has no {METRIC_TO_MAXIMIZE} metric.")
            return False
        promote = candidate_score > champion_score
        print(f"[INFO] Candidate R2={candidate_score:.4f}; champion R2={champion_score:.4f}.")

    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", str(candidate.version))
        print(f"[SUCCESS] Version {candidate.version} is now champion.")
    else:
        print(f"[INFO] Candidate version {candidate.version} did not beat the champion; champion retained.")

    client.delete_registered_model_alias(MODEL_NAME, "candidate")
    return True


if __name__ == "__main__":
    sys.exit(0 if promote_candidate() else 1)
