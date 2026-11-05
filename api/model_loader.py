import json
from pathlib import Path

import joblib
import numpy
import sklearn


class ModelLoader:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or Path(__file__).resolve().parents[1] / "deployment"
        self.model = None
        self.preprocessor = None
        self.metadata = None

    def load(self) -> None:
        model_path = self.model_dir / "model.pkl"
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        metadata_path = self.model_dir / "model_metadata.json"
        for path in (model_path, preprocessor_path, metadata_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Deployment artifact is missing: {path}. Run Lab 10 before starting the API."
                )

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_version = metadata.get("scikit_learn_version")
        if saved_version != sklearn.__version__:
            raise RuntimeError(
                f"Model was created with scikit-learn {saved_version}, but runtime has "
                f"{sklearn.__version__}. Use the matching requirements-api.txt."
            )
        saved_numpy_version = metadata.get("numpy_version")
        if saved_numpy_version != numpy.__version__:
            raise RuntimeError(
                f"Model was created with NumPy {saved_numpy_version}, but runtime has "
                f"{numpy.__version__}. Use the matching requirements-api.txt."
            )

        required_transformers = {"imputer", "scaler", "ohe", "number_of_bags_cap"}
        missing_transformers = required_transformers - set(preprocessor)
        if missing_transformers:
            raise ValueError(
                f"Preprocessor artifact is missing required values: {sorted(missing_transformers)}"
            )
        expected_features = metadata.get("feature_count")
        if expected_features is None or getattr(model, "n_features_in_", None) != expected_features:
            raise ValueError("Model feature count does not match deployment metadata.")

        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata

    def predict_records(self, records: list[dict]) -> list[float]:
        if self.model is None or self.preprocessor is None or self.metadata is None:
            raise RuntimeError("The coffee prediction model has not been loaded.")

        from api.preprocessing import transform_records

        predictions = self.model.predict(transform_records(records, self.preprocessor))
        return [float(prediction) for prediction in predictions]
