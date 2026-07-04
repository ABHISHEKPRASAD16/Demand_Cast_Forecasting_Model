"""Great Expectations gate for fct_sales, run right after it's loaded from
the warehouse and before any feature engineering or training.

This checks something different from the dbt tests in
dbt/demandcast/models/marts/_marts.yml (Phase 1): those validate the
warehouse is well-formed once, at build time. This validates that the exact
DataFrame a training run is about to consume still looks sane at run time -
training code shouldn't have to assume dbt was recently and correctly
rerun, or that nothing upstream silently changed.
"""

import great_expectations as gx
import pandas as pd
from great_expectations.core.expectation_validation_result import ExpectationSuiteValidationResult


class DataValidationError(Exception):
    """Raised when fct_sales fails its expectation suite before training starts."""


def _build_fct_sales_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="fct_sales_suite")
    for expectation in (
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="store_id"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="sale_date"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="sales"),
        gx.expectations.ExpectColumnValuesToBeBetween(column="sales", min_value=0),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="day_of_week", min_value=1, max_value=7
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(column="is_open", value_set=[True]),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="store_type", value_set=["a", "b", "c", "d"]
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(column="assortment", value_set=["a", "b", "c"]),
        # nulls are allowed here (some stores have no recorded competitor)
        # - this only fires on a non-null negative distance
        gx.expectations.ExpectColumnValuesToBeBetween(column="competition_distance", min_value=0),
    ):
        suite.add_expectation(expectation)
    return suite


def validate_fct_sales(df: pd.DataFrame) -> ExpectationSuiteValidationResult:
    """Run the fct_sales expectation suite and return the full result (never raises)."""
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="fct_sales")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    return batch.validate(_build_fct_sales_suite())


def validate_fct_sales_or_raise(df: pd.DataFrame) -> None:
    """Convenience wrapper for training entrypoints: fail fast and loudly."""
    result = validate_fct_sales(df)
    if not result.success:
        failed = [r.expectation_config.type for r in result.results if not r.success]
        raise DataValidationError(f"fct_sales failed expectations: {failed}")
