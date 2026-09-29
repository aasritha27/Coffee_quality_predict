"""Lab 4: MLflow-tracked training. One meaningful change per run; seed fixed."""
import argparse, json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import *

p = argparse.ArgumentParser()
p.add_argument("--model", default="rf", choices=["rf", "gbr", "ridge"])
p.add_argument("--n_estimators", type=int, default=100)
p.add_argument("--max_depth", type=int, default=10)
p.add_argument("--min_samples_leaf", type=int, default=1)
p.add_argument("--alpha", type=float, default=1.0, help="ridge only")
p.add_argument("--random_state", type=int, default=SEED)
p.add_argument("--run_name", default=None)
a = p.parse_args()

def build():
    if a.model == "rf":
        return RandomForestRegressor(n_estimators=a.n_estimators, max_depth=a.max_depth,
                                     min_samples_leaf=a.min_samples_leaf,
                                     random_state=a.random_state, n_jobs=-1)
    if a.model == "gbr":
        return GradientBoostingRegressor(n_estimators=a.n_estimators, max_depth=a.max_depth,
                                         random_state=a.random_state)
    return Ridge(alpha=a.alpha)

Xtr, ytr = np.load(PROC / "X_train_final.npy"), np.load(PROC / "y_train.npy")
Xte, yte = np.load(PROC / "X_test_final.npy"), np.load(PROC / "y_test.npy")
meta = json.loads((PROC / "dataset_metadata.json").read_text())

# MLflow >= 3.x deprecates the plain ./mlruns file store, so metadata goes to SQLite
# (mlflow.db) and artifacts (plots, model) are stored under ./mlruns.
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
EXPERIMENT = "Coffee_Quality_Prediction"
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
mlflow.set_experiment(EXPERIMENT)
name = a.run_name or f"{a.model}_n{a.n_estimators}_d{a.max_depth}_leaf{a.min_samples_leaf}"
ARTIFACTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)

with mlflow.start_run(run_name=name):
    model = build().fit(Xtr, ytr)
    pred = model.predict(Xte)
    params = {"model_type": type(model).__name__, "random_state": a.random_state,
              **{k: v for k, v in model.get_params().items()
                 if k in ("n_estimators", "max_depth", "min_samples_leaf", "alpha")}}
    mlflow.log_params(params)
    metrics = {"mae": mean_absolute_error(yte, pred), "rmse": mean_squared_error(yte, pred) ** 0.5,
               "r2": r2_score(yte, pred), "train_r2": model.score(Xtr, ytr)}
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"task": "regression", "target": TARGET, "dataset": "CQI merged"})

    # plots
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yte, pred, alpha=.5); lo, hi = yte.min(), yte.max()
    ax.plot([lo, hi], [lo, hi], "r--"); ax.set(xlabel="Actual", ylabel="Predicted",
                                              title=f"Predicted vs Actual (R2={metrics['r2']:.3f})")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "predicted_vs_actual.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pred - yte, bins=40); ax.set(xlabel="Prediction error", title="Residuals"); fig.tight_layout()
    fig.savefig(ARTIFACTS / "residuals.png", dpi=130); plt.close(fig)
    for f in ("predicted_vs_actual.png", "residuals.png"):
        mlflow.log_artifact(str(ARTIFACTS / f))
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=meta["feature_names"]).nlargest(15)[::-1]
        fig, ax = plt.subplots(figsize=(7, 5)); imp.plot.barh(ax=ax); ax.set_title("Top 15 feature importances")
        fig.tight_layout(); fig.savefig(ARTIFACTS / "feature_importance.png", dpi=130); plt.close(fig)
        mlflow.log_artifact(str(ARTIFACTS / "feature_importance.png"))
    mlflow.log_artifact(str(PROC / "dataset_metadata.json"))
    np.save(ARTIFACTS / "test_predictions.npy", pred)     # for prediction-level reproducibility checks
    mlflow.log_artifact(str(ARTIFACTS / "test_predictions.npy"))
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle",
                             signature=infer_signature(Xte[:5], pred[:5]))
    joblib.dump(model, MODELS / "random_forest_model.pkl")   # local backup
    print(name, {k: round(v, 4) for k, v in metrics.items()})
