# Coffee Quality Prediction — MLOps Labs 1–6

Regression project on the Coffee Quality Institute (CQI) data
(<https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi>).
Target: `Total.Cup.Points`. Same lab structure as the Telco churn guide (preprocess → baseline → Git → MLflow).

## Setup (from the project root)
```bash
python -m venv .venv                 # Windows: py -3.11 -m venv .venv
source .venv/bin/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run order
```bash
python src/merge_data.py             # (only if you re-download arabica/robusta files) -> data/raw/merged_data.csv
python src/preprocess.py             # Lab 1  -> data/processed/*.npy, models/{imputer,scaler,ohe}.pkl, metadata
python src/train.py                  # Lab 2  -> models/random_forest_baseline.pkl
python src/evaluate.py               # Lab 2  -> outputs/*.csv, baseline_metrics.json
bash scripts/git_workflow_lab3.sh    # Lab 3  -> branches, conflict, revert, tag (Git Bash / Linux / macOS)
python src/run_experiments.py        # Lab 4  -> 8 tracked runs
python src/check_reproducibility.py  # Lab 4  -> same config twice, predictions compared
python scripts/run_lab5_pipeline.py  # Lab 5  -> validate data, preprocess, validate outputs, reproduce
python scripts/run_lab6_registry.py  # Lab 6  -> register candidate, promote champion, report deployment info
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1 --host 127.0.0.1
```
MLflow ≥ 3 no longer supports the plain `./mlruns` file store, so run metadata is in `mlflow.db`
and artifacts (plots, model) are under `mlruns/`. Both are git-ignored and created by your own runs.
The Lab 6 registry uses the `candidate` and `champion` model aliases and promotes a candidate only
when its test-set R² is higher. On Windows, `--workers 1` avoids multi-worker socket startup failures.

## Key design decisions
- **Leakage guard:** the 10 sensory cup scores sum to `Total.Cup.Points`; they are excluded so the model
  learns from origin/processing/altitude/defects. (EDA notebook §6 shows the proof.)
- **Cleaning:** drops one 0-point lot; altitude > 5000 m and moisture > 0.5 set to missing; bag weight (kg) and harvest year parsed from text; median/“Unknown” imputation fit on train only.
- **Split:** 80/20, `random_state=42`, stratified on target quintiles.

## Results (test set, 268 rows)
| Run | MAE | RMSE | R² |
|---|---|---|---|
| mean-prediction dummy | 1.96 | – | 0.00 |
| rf_baseline (100 trees, depth 10) | 1.651 | 2.629 | 0.261 |
| rf_trees_50 | 1.643 | 2.611 | 0.271 |
| rf_trees_300 | 1.657 | 2.637 | 0.257 |
| rf_depth_5 | 1.736 | 2.752 | 0.190 |
| rf_depth_20 | 1.627 | 2.608 | 0.273 |
| rf_leaf_3 | 1.667 | 2.669 | 0.238 |
| gbr_n200_d3 | 1.702 | 2.654 | 0.247 |
| ridge_alpha10 | 1.699 | 2.683 | 0.230 |

Repeated identical runs give identical metrics and bit-identical predictions.
Modest R² is expected: without tasting scores, metadata explains only part of quality.
To include the sensory scores (R² ≈ 0.99, but leaky), add `SENSORY` to `NUMERIC` in `src/common.py`.

## Evidence for your report (Section 14 of the guide)
Lab 1: notebook + `dataset_metadata.json` · Lab 2: `outputs/` · Lab 3: `git log --oneline --graph --all`, `git tag` ·
Lab 4: MLflow UI screenshots (compare runs, artifacts) + reproducibility output.
