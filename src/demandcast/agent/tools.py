"""Tools the agent can call: retrieval over the RAG knowledge base, and a
live forecast against the DemandCast API. Scoping tools this narrowly is
itself a security measure - a full prompt-injection jailbreak still can't
do anything worse than get a wrong forecast number or an off-topic reply,
since the agent has no shell, filesystem, or general-network tool available
to it, only a pydantic-validated call to our own /predict endpoint.
"""

import httpx
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field

from demandcast.agent.vectorstore import load_vectorstore

DEFAULT_API_BASE_URL = "http://localhost:8000"
RETRIEVER_K = 4


class ForecastToolInput(BaseModel):
    store_id: int = Field(..., gt=0, description="Rossmann store id")
    date: str = Field(..., description="Date to forecast, as YYYY-MM-DD")
    is_promo: bool = Field(False, description="Simulate a promo running on this date")
    is_school_holiday: bool = Field(False, description="Simulate a school holiday on this date")
    state_holiday: str = Field("0", description="State holiday code: '0', 'a', 'b', or 'c'")


def build_forecast_tool(api_base_url: str = DEFAULT_API_BASE_URL):
    @tool("forecast_sales", args_schema=ForecastToolInput)
    def forecast_sales(
        store_id: int,
        date: str,
        is_promo: bool = False,
        is_school_holiday: bool = False,
        state_holiday: str = "0",
    ) -> str:
        """Call the live DemandCast forecast API for one store and date - use this
        for any 'what if' scenario question (e.g. running a promo) rather than
        guessing a number."""
        try:
            response = httpx.post(
                f"{api_base_url}/predict",
                json={
                    "store_id": store_id,
                    "date": date,
                    "is_promo": is_promo,
                    "is_school_holiday": is_school_holiday,
                    "state_holiday": state_holiday,
                },
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", exc.response.text)
            return f"Forecast API returned an error ({exc.response.status_code}): {detail}"
        except httpx.RequestError as exc:
            return f"Could not reach the forecast API: {exc}"

        data = response.json()
        return (
            f"Predicted sales for store {data['store_id']} on {data['date']}: "
            f"{data['predicted_sales']:,.0f} (model {data['model_name']} v{data['model_version']})"
        )

    return forecast_sales


def build_retriever_tool():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    return create_retriever_tool(
        retriever,
        name="search_demandcast_knowledge",
        description=(
            "Search DemandCast's own documentation (metric definitions, model design "
            "decisions, architecture) and per-store monthly sales summaries. Use this "
            "before answering any question about metrics, why a model was chosen, or "
            "historical sales patterns for a specific store or month."
        ),
    )
