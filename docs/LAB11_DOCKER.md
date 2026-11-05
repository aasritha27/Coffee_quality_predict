# Lab 11 — Docker prediction API

After Lab 10, build and run the coffee prediction API:

```powershell
python pipelines\run_lab11_container.py
docker run --rm -p 8000:8000 coffee-quality-mlops:local
```

The container exposes `GET /health`, `POST /predict`, and `POST /predict/batch`. Swagger UI is
available at `http://localhost:8000/docs`. The API accepts only coffee-quality fields and predicts
`Total.Cup.Points`; it does not use customer/churn fields.
