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
