import pandas as pd

from demandcast.agent.knowledge import _month_summary, load_static_documents


def test_load_static_documents_reads_all_knowledge_markdown_files():
    documents = load_static_documents()

    sources = {doc.metadata["source"] for doc in documents}
    assert sources == {"architecture.md", "metrics.md", "models.md"}
    assert all(len(doc.page_content) > 0 for doc in documents)


def _synthetic_month(sales_values: list[float], promo_days: int = 2) -> pd.DataFrame:
    n = len(sales_values)
    return pd.DataFrame(
        {
            "sales": sales_values,
            "is_promo": [True] * promo_days + [False] * (n - promo_days),
            "is_school_holiday": [False] * n,
            "state_holiday": ["0"] * n,
            "store_type": ["a"] * n,
            "assortment": ["a"] * n,
        }
    )


def test_month_summary_includes_key_facts():
    group = _synthetic_month([100.0, 200.0, 300.0])
    period = pd.Period("2015-03", freq="M")

    summary = _month_summary(store_id=262, period=period, group=group, prior_avg=None)

    assert "Store 262" in summary
    assert "2015-03" in summary
    assert "2 promo day" in summary
    assert "average daily sales 200" in summary


def test_month_summary_reports_mom_change_direction():
    group = _synthetic_month([100.0, 100.0, 100.0], promo_days=0)
    period = pd.Period("2015-04", freq="M")

    down_summary = _month_summary(store_id=1, period=period, group=group, prior_avg=200.0)
    up_summary = _month_summary(store_id=1, period=period, group=group, prior_avg=50.0)

    assert "down 50.0%" in down_summary
    assert "up 100.0%" in up_summary


def test_month_summary_omits_mom_change_without_prior_month():
    group = _synthetic_month([100.0], promo_days=0)
    summary = _month_summary(
        store_id=1, period=pd.Period("2015-01", freq="M"), group=group, prior_avg=None
    )
    assert "versus the previous month" not in summary
