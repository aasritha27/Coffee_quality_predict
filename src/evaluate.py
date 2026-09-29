"""Lab 2: evaluate baseline on the test set; export error-analysis CSVs."""
import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import *

X, y = np.load(PROC / "X_test_final.npy"), np.load(PROC / "y_test.npy")
model = joblib.load(MODELS / "random_forest_baseline.pkl")
pred = model.predict(X)
m = {"mae": mean_absolute_error(y, pred), "rmse": mean_squared_error(y, pred) ** 0.5,
     "r2": r2_score(y, pred)}
# naive baseline: always predict the training mean
ytr = np.load(PROC / "y_train.npy")
m["dummy_mae"] = mean_absolute_error(y, np.full_like(y, ytr.mean()))
print({k: round(v, 4) for k, v in m.items()})

rows = pd.read_csv(PROC / "test_rows_clean.csv")
rows["predicted"], rows["error"] = pred, pred - y
OUTPUTS.mkdir(exist_ok=True)
cols = ["Species", "Country.of.Origin", "Variety", "Processing.Method", TARGET, "predicted", "error"]
rows.nlargest(25, "error")[cols].to_csv(OUTPUTS / "worst_overpredictions.csv", index=False)
rows.nsmallest(25, "error")[cols].to_csv(OUTPUTS / "worst_underpredictions.csv", index=False)
(OUTPUTS / "baseline_metrics.json").write_text(json.dumps(m, indent=2))
print("Saved outputs/worst_overpredictions.csv, worst_underpredictions.csv, baseline_metrics.json")
