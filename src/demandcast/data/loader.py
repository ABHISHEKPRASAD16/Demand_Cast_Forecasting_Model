"""Read the modeling table (marts.fct_sales) out of the DuckDB warehouse dbt builds."""

import duckdb
import pandas as pd

from demandcast.config import load_config


def load_fct_sales(store_id: int | None = None, open_only: bool = True) -> pd.DataFrame:
    """Load fct_sales, optionally filtered to one store and/or open days only.

    `open_only` defaults to True because sales is deterministically 0 on days a
    store is closed - including those rows would make every model look
    artificially better at "predicting" zeros without teaching it anything
    about actual demand.
    """
    config = load_config()
    con = duckdb.connect(str(config.warehouse.duckdb_path), read_only=True)

    conditions = []
    params: list[int] = []
    if store_id is not None:
        conditions.append("store_id = ?")
        params.append(store_id)
    if open_only:
        conditions.append("is_open = true")
    where_sql = f"where {' and '.join(conditions)}" if conditions else ""

    query = f"""
        select *
        from {config.warehouse.marts_schema}.fct_sales
        {where_sql}
        order by store_id, sale_date
    """
    df = con.execute(query, params).df()
    con.close()
    return df


def list_store_ids() -> list[int]:
    """All distinct store ids in fct_sales - cheap enough to not need load_fct_sales'
    full-row fetch just to enumerate them (used by the dashboard's store picker)."""
    config = load_config()
    con = duckdb.connect(str(config.warehouse.duckdb_path), read_only=True)
    ids = con.execute(
        f"select distinct store_id from {config.warehouse.marts_schema}.fct_sales order by store_id"
    ).df()["store_id"]
    con.close()
    return ids.tolist()


def get_sale_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """(min, max) sale_date across the whole warehouse - used to keep the
    dashboard's forecast date picker within a range where history actually
    exists to compute lag/rolling features from."""
    config = load_config()
    con = duckdb.connect(str(config.warehouse.duckdb_path), read_only=True)
    row = con.execute(
        f"select min(sale_date), max(sale_date) from {config.warehouse.marts_schema}.fct_sales"
    ).fetchone()
    con.close()
    return pd.Timestamp(row[0]), pd.Timestamp(row[1])
