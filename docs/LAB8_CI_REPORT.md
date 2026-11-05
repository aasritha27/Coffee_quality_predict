# Lab 8 — CI and model quality gate

The GitHub Actions workflow at `.github/workflows/lab8_ci.yml` installs project dependencies,
runs unit tests, validates the coffee source data, and evaluates the baseline quality gate.
The gate requires finite regression metrics and an MAE lower than the training-mean dummy baseline.
Its machine-readable result is written to `reports/quality_gate_report.json`.

Run locally with `python pipelines\run_lab8_quality_gate.py`.
