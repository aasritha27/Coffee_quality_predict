# Lab 10 — Champion deployment preparation

`python pipelines\run_lab10_deployment.py` trains/promotes the coffee registry candidate, runs the
quality gate, and exports the `champion` model and the corresponding coffee preprocessors into
`deployment/`. The deployment metadata records the target, model version, feature count, preprocessing
cap, and scikit-learn version. The API refuses to load artifacts from a different scikit-learn version.
