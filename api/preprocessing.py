from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from common import CATEGORICAL, NUMERIC, TARGET
from preprocess import clean


def transform_records(records: list[dict], preprocessor: dict) -> np.ndarray:
    if not records:
        raise ValueError("At least one coffee record is required.")

    frame = pd.DataFrame.from_records(records)
    frame[TARGET] = 1.0
    for column in NUMERIC:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    cleaned = clean(
        frame,
        number_of_bags_cap=preprocessor["number_of_bags_cap"],
        verbose=False,
    )
    numeric = preprocessor["scaler"].transform(
        preprocessor["imputer"].transform(cleaned[NUMERIC])
    )
    categorical = preprocessor["ohe"].transform(cleaned[CATEGORICAL])
    features = np.hstack([numeric, categorical])
    if not np.isfinite(features).all():
        raise ValueError("Preprocessing produced non-finite feature values.")
    return features
