"""Builds the RAG knowledge base: static docs (metrics/model/architecture
explanations) plus dynamic per-store-month insight blurbs computed from
fct_sales.

Dynamic docs are generated for a curated set of stores rather than all
1,115 - the flagship store (262, see Phase 2) plus the next 9 highest-
volume stores with complete open-day history. That covers enough range
for demo questions without spending minutes re-embedding a full-panel
corpus on every rebuild; scaling to all stores is just widening
DEMO_STORE_IDS.
"""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from demandcast.data import load_fct_sales

KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "knowledge"

DEMO_STORE_IDS = [262, 562, 733, 335, 682, 423, 769, 1097, 494, 85]


def load_static_documents() -> list[Document]:
    documents = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        documents.append(Document(page_content=path.read_text(), metadata={"source": path.name}))
    return documents


def _month_summary(
    store_id: int, period: pd.Period, group: pd.DataFrame, prior_avg: float | None
) -> str:
    avg_sales = group["sales"].mean()
    total_sales = group["sales"].sum()
    promo_days = int(group["is_promo"].sum())
    school_holiday_days = int(group["is_school_holiday"].sum())
    state_holiday_days = int((group["state_holiday"] != "0").sum())
    store_type = group["store_type"].iloc[0]
    assortment = group["assortment"].iloc[0]

    mom_change = ""
    if prior_avg is not None and prior_avg > 0:
        pct = (avg_sales - prior_avg) / prior_avg * 100
        direction = "up" if pct >= 0 else "down"
        mom_change = (
            f" Average daily sales were {direction} {abs(pct):.1f}% versus the previous month."
        )

    return (
        f"Store {store_id} ({period}): total sales {total_sales:,.0f}, "
        f"average daily sales {avg_sales:,.0f} across {len(group)} open days. "
        f"{promo_days} promo day(s), {school_holiday_days} school holiday day(s), "
        f"{state_holiday_days} state holiday day(s). "
        f"Store type '{store_type}', assortment '{assortment}'.{mom_change}"
    )


def build_store_month_documents(store_ids: list[int] = DEMO_STORE_IDS) -> list[Document]:
    documents = []
    for store_id in store_ids:
        df = load_fct_sales(store_id=store_id)
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        df["year_month"] = df["sale_date"].dt.to_period("M")

        prior_avg: float | None = None
        for period, group in df.groupby("year_month"):
            documents.append(
                Document(
                    page_content=_month_summary(store_id, period, group, prior_avg),
                    metadata={
                        "source": f"store_{store_id}_{period}",
                        "store_id": store_id,
                        "year_month": str(period),
                    },
                )
            )
            prior_avg = float(group["sales"].mean())
    return documents


def build_knowledge_base(store_ids: list[int] = DEMO_STORE_IDS) -> list[Document]:
    return load_static_documents() + build_store_month_documents(store_ids)
