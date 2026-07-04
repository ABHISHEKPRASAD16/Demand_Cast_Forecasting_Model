from datetime import date as Date

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    store_id: int = Field(..., gt=0, description="Rossmann store id")
    date: Date = Field(..., description="Date to forecast sales for")
    is_promo: bool = Field(False, description="Is a promo running on this date?")
    is_school_holiday: bool = Field(False, description="Is this date a school holiday?")
    state_holiday: str = Field("0", description="State holiday code: '0','a','b','c'")


class PredictResponse(BaseModel):
    store_id: int
    date: Date
    predicted_sales: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelMetadataResponse(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    metrics: dict[str, float]
