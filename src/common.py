"""Shared constants/helpers. Always run scripts from the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "merged_data.csv"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"
TARGET = "Total.Cup.Points"
SEED = 42
RF_PARAMS = {
    "n_estimators": 500,
    "max_features": 0.8,
    "min_samples_leaf": 2,
    "random_state": SEED,
    "n_jobs": -1,
}

# The 10 cup-score columns sum (almost exactly) to Total.Cup.Points, so using them
# to predict it is target leakage. The prediction feature set therefore excludes
# these columns and uses origin, processing, altitude, defects, and other metadata.
SENSORY = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
           "Uniformity", "Clean.Cup", "Sweetness", "Cupper.Points"]
NUMERIC = ["Number.of.Bags", "Bag.Weight.kg", "Harvest.Year.num", "Moisture",
           "Category.One.Defects", "Quakers", "Category.Two.Defects",
           "altitude_mean_meters"]
CATEGORICAL = ["Species", "Country.of.Origin", "Variety", "Processing.Method", "Color"]
