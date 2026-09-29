"""Lab 2: baseline Random Forest regressor on processed arrays."""
import joblib, numpy as np
from sklearn.ensemble import RandomForestRegressor
from common import *

X, y = np.load(PROC / "X_train_final.npy"), np.load(PROC / "y_train.npy")
model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
model.fit(X, y)
MODELS.mkdir(exist_ok=True)
joblib.dump(model, MODELS / "random_forest_baseline.pkl")
print("Saved models/random_forest_baseline.pkl | train R2 = %.4f" % model.score(X, y))
