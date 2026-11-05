from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.model_loader import ModelLoader
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CoffeeInput,
    PredictionResponse,
)


def create_app(model_loader: ModelLoader | None = None) -> FastAPI:
    loader = model_loader or ModelLoader()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader.load()
        app.state.model_loader = loader
        yield

    application = FastAPI(
        title="Coffee Quality Prediction API",
        version="1.0.0",
        description="Predicts the coffee quality score from non-sensory CQI features.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": loader.metadata["model_version"]}

    @application.post("/predict", response_model=PredictionResponse)
    def predict(record: CoffeeInput) -> PredictionResponse:
        prediction = loader.predict_records([record.model_dump(by_alias=True)])[0]
        return PredictionResponse(
            prediction=prediction,
            target=loader.metadata["target"],
            model_version=loader.metadata["model_version"],
        )

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
        records = [record.model_dump(by_alias=True) for record in request.records]
        return BatchPredictionResponse(
            predictions=loader.predict_records(records),
            target=loader.metadata["target"],
            model_version=loader.metadata["model_version"],
        )

    return application


app = create_app()
