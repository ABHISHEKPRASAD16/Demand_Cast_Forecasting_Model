"""FastAPI service: predict / model metadata / health.

uvicorn demandcast.serving.main:app --reload
"""

import logging
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from demandcast.serving.inference import (
    HistoryFetcher,
    get_history_fetcher,
    get_model,
    get_model_metadata,
    get_model_version,
    predict_sales,
)
from demandcast.serving.schemas import (
    HealthResponse,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="DemandCast API", version="0.1.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, duration_ms
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata(metadata: dict = Depends(get_model_metadata)) -> ModelMetadataResponse:
    return ModelMetadataResponse(**metadata)


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    model=Depends(get_model),
    model_version: str = Depends(get_model_version),
    history_fetcher: HistoryFetcher = Depends(get_history_fetcher),
) -> PredictResponse:
    history = history_fetcher(request.store_id, request.date)
    if history.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No sales history found for store_id={request.store_id} before {request.date}",
        )

    try:
        prediction = predict_sales(model, history, request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(
        store_id=request.store_id,
        date=request.date,
        predicted_sales=prediction,
        model_name="demandcast-lightgbm",
        model_version=model_version,
    )
