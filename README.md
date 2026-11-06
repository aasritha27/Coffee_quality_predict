# Coffee Quality Prediction — MLOps Labs 1–13

Regression project on the Coffee Quality Institute (CQI) data
(<https://www.kaggle.com/datasets/volpatto/coffee-quality-database-from-cqi>).
Target: `Total.Cup.Points`. The churn-project ZIP is used only as a layout and workflow reference; all data,
features, models, and API behavior here are coffee-quality-specific.

## Setup (PowerShell, from the project root)

The project is inside the `coffee-quality-mlops` subfolder. From the parent folder, enter it first:
```powershell
Set-Location .\coffee-quality-mlops
```

If this project already has a `.venv`, activate it instead of creating it again:
```powershell
.\.venv\Scripts\Activate.ps1
```

For a fresh environment, Python 3.13 is available through the Windows launcher on this machine:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies:
```powershell
python -m pip install -r requirements.txt
```

## Labs 1–6
```powershell
python src\merge_data.py             # Optional: only after re-downloading source CSVs
python src\preprocess.py             # Lab 1: preprocess, split, and save reusable transformers
python src\train.py                  # Lab 2: train baseline
python src\evaluate.py               # Lab 2: metrics and error-analysis CSVs
bash scripts/git_workflow_lab3.sh   # Lab 3: Git exercise (Git Bash required; commits and rewrites train.py)
python src\run_experiments.py        # Lab 4: eight MLflow runs
python src\check_reproducibility.py  # Lab 4: deterministic training check
python scripts\run_lab5_pipeline.py  # Lab 5: validation/preprocessing/reproducibility pipeline
python scripts\run_lab6_registry.py  # Lab 6: candidate/champion registry lifecycle
```

Lab 3 is an intentional Git exercise: it creates commits, branches, a merge conflict, a revert, and a tag.
Run it only in a disposable clone or after saving/committing your own work.

## Labs 7–11
```powershell
python pipelines\run_lab7_pipeline.py      # Lab 7: end-to-end coffee pipeline
python pipelines\run_lab8_quality_gate.py  # Lab 8: metrics quality gate
python pipelines\run_lab9_failure_tests.py # Lab 9: safe failure-injection tests
python pipelines\run_lab10_deployment.py   # Lab 10: promote/report/export the champion
python pipelines\run_lab11_container.py   # Lab 11: build the prediction API image
```

## Labs 12–13

Lab 12 compares the saved training and holdout feature/prediction distributions with a
two-sample Kolmogorov–Smirnov test, reports holdout regression metrics, and writes
`reports/drift_monitoring_report.json`. Run it after Labs 1–2:
```powershell
python pipelines\run_lab12_monitoring.py
```

Lab 13 runs the validated pipeline, quality gate, champion deployment preparation, Lab 12
monitoring, and integration tests against the prepared production API model. Its stage
results are written to `logs/lab13_end_to_end_run_log.json`:
```powershell
python pipelines\run_lab13_end_to_end.py
```

The monitor compares the project’s saved training and test splits; it does not ingest an
external production batch. A drift report is evidence for review, not an automatic promotion
or retraining decision.

The Lab 11 API serves coffee score predictions from the stored champion model:
```powershell
docker run --rm -p 8000:8000 coffee-quality-mlops:local
```
In another terminal:
```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body (@{
  species='Arabica'; country_of_origin='Ethiopia'; variety='Heirloom'; processing_method='Washed / Wet'
  color='Green'; number_of_bags=300; bag_weight='60 kg'; harvest_year='2024'
  moisture=0.12; category_one_defects=0; quakers=0; category_two_defects=0; altitude_mean_meters=1800
} | ConvertTo-Json)
```

Direct Docker commands, after completing Lab 10:
```powershell
docker build -t coffee-quality-mlops:local .
docker run --rm -p 8000:8000 coffee-quality-mlops:local
```

## MLflow UI
```powershell
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

Repeated identical runs give identical metrics and predictions within a 1e-12 absolute tolerance
(parallel tree prediction can differ at floating-point rounding precision).
Modest R² is expected: without tasting scores, metadata explains only part of quality.
To include the sensory scores (R² ≈ 0.99, but leaky), add `SENSORY` to `NUMERIC` in `src/common.py`.

## Evidence for your report (Section 14 of the guide)
Lab 1: notebook + `dataset_metadata.json` · Lab 2: `outputs/` · Lab 3: `git log --oneline --graph --all`, `git tag` ·
Lab 4: MLflow UI screenshots (compare runs, artifacts) + reproducibility output · Labs 7–11:
`reports/`, API tests, `deployment/model_metadata.json`, and the running Docker API.
