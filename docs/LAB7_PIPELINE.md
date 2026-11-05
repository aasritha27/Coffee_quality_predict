# Lab 7 — End-to-end coffee pipeline

Run `python pipelines\run_lab7_pipeline.py` from the project root. It validates and preprocesses
the current CQI coffee data, checks processed outputs and reproducibility, trains and evaluates the
baseline, and runs the existing coffee MLflow registry pipeline. It stops on the first failed stage.

The pipeline uses `Total.Cup.Points` as the regression target and the non-sensory features already
defined in `src/common.py`; it does not import churn data or churn model artifacts.
