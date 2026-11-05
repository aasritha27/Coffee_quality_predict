import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import CATEGORICAL, NUMERIC, PROC, RAW
from registry_utils import MODEL_NAME, configure_tracking


def prepare_deployment() -> None:
    configure_tracking()
    client = MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run = client.get_run(champion.run_id)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    dataset_metadata = json.loads((PROC / "dataset_metadata.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(RAW)
    raw = raw.loc[pd.to_numeric(raw["Total.Cup.Points"], errors="coerce") > 0]
    bag_count_cap = float(raw["Number.of.Bags"].quantile(0.99))
    expected_features = len(NUMERIC) + len(
        joblib.load(ROOT / "models" / "ohe.pkl").get_feature_names_out(CATEGORICAL)
    )
    if getattr(model, "n_features_in_", None) != expected_features:
        raise ValueError(
            f"Champion expects {getattr(model, 'n_features_in_', None)} features; "
            f"current coffee preprocessing produces {expected_features}."
        )

    deployment_dir = ROOT / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, deployment_dir / "model.pkl")
    joblib.dump(
        {
            "imputer": joblib.load(ROOT / "models" / "imputer.pkl"),
            "scaler": joblib.load(ROOT / "models" / "scaler.pkl"),
            "ohe": joblib.load(ROOT / "models" / "ohe.pkl"),
            "number_of_bags_cap": bag_count_cap,
        },
        deployment_dir / "preprocessor.pkl",
    )

    metadata = {
        "target": dataset_metadata["target"],
        "dataset": dataset_metadata["dataset"],
        "feature_count": expected_features,
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": numpy.__version__,
        "model_version": str(champion.version),
        "registry_name": MODEL_NAME,
        "run_id": champion.run_id,
        "metrics": run.data.metrics,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "number_of_bags_cap": bag_count_cap,
    }
    (deployment_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"[SUCCESS] Exported {MODEL_NAME} version {champion.version} "
        f"with {expected_features} features to deployment/."
    )


if __name__ == "__main__":
    prepare_deployment()
