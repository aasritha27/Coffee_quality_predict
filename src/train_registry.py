"""Lab 6: train, evaluate, and register a coffee-quality regression model."""
import json

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import MODELS, PROC, SEED
from registry_utils import MODEL_NAME, configure_tracking


def train_and_register_model() -> str:
    configure_tracking()
    X_train = np.load(PROC / "X_train_final.npy")
    X_test = np.load(PROC / "X_test_final.npy")
    y_train = np.load(PROC / "y_train.npy")
    y_test = np.load(PROC / "y_test.npy")
    metadata = json.loads((PROC / "dataset_metadata.json").read_text(encoding="utf-8"))

    params = {"n_estimators": 200, "max_depth": 15, "random_state": SEED, "n_jobs": -1}
    client = MlflowClient()
    with mlflow.start_run(run_name="coffee_random_forest_candidate") as run:
        model = RandomForestRegressor(**params).fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }

        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForestRegressor")
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "task": "regression",
            "target": metadata["target"],
            "dataset": metadata["dataset"],
        })

        for filename in ("imputer.pkl", "scaler.pkl", "ohe.pkl"):
            path = MODELS / filename
            if not path.is_file():
                raise FileNotFoundError(f"Preprocessing artifact is missing: {path}. Run Lab 5 first.")
            mlflow.log_artifact(str(path), artifact_path="preprocessing")
        mlflow.log_artifact(str(PROC / "dataset_metadata.json"), artifact_path="preprocessing")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            signature=infer_signature(X_test[:5], predictions[:5]),
            input_example=X_test[:5],
            serialization_format="cloudpickle",
        )

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        run_versions = [version for version in versions if version.run_id == run.info.run_id]
        if not run_versions:
            raise RuntimeError("MLflow logged the model but did not create a registry version.")
        candidate = max(run_versions, key=lambda version: int(version.version))
        client.set_registered_model_alias(MODEL_NAME, "candidate", str(candidate.version))

    print(f"[SUCCESS] Registered candidate version {candidate.version} of {MODEL_NAME}.")
    print("Metrics:", {key: round(value, 4) for key, value in metrics.items()})
    return str(candidate.version)


if __name__ == "__main__":
    train_and_register_model()
