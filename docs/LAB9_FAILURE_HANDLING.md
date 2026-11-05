# Lab 9 — Failure handling

`python pipelines\run_lab9_failure_tests.py` tests a missing raw dataset and a CSV missing required
coffee columns. Both cases must fail validation and write an explicit failure report. Test inputs and
reports live in temporary directories, so the project dataset and tracked artifacts are not changed.
