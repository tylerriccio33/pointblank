from __future__ import annotations

import pytest
import narwhals as nw

from hypothesis import given, settings, strategies as st
import polars.testing.parametric as pt
from great_tables import GT
from typing import TYPE_CHECKING
import polars as pl

from pointblank.datascan import DataScan, col_summary_tbl
from pointblank._datascan_utils import _compact_0_1_fmt, _compact_decimal_fmt, _compact_integer_fmt
from pointblank.scan_profile import _as_physical
from pointblank.scan_profile_stats import StatGroup, COLUMN_ORDER_REGISTRY

if TYPE_CHECKING:
    import pyarrow as pa
    import pandas as pd


## Setup Strategies:
## Generate df and ldf happy paths using polars.
## Also generate pandas and arrow strategies which should smoke test any complete mistakes
## or inconsistent handling in narwhals. Really checking the consistency among packages is
## too much the job of narwhals, and we should avoid stepping on their testing suite.
happy_path_df = pt.dataframes(
    min_size=5,
    allowed_dtypes=[pl.Int64, pl.Float64, pl.String, pl.Categorical, pl.Date, pl.Datetime],
)
happy_path_ldf = pt.dataframes(
    min_size=5,
    allowed_dtypes=[pl.Int64, pl.Float64, pl.String, pl.Categorical, pl.Date, pl.Datetime],
    lazy=True,
)


@st.composite
def _arrow_strat(draw) -> pa.Table:
    polars_df = draw(happy_path_df)
    return nw.from_native(polars_df).to_arrow()


@st.composite
def _pandas_strat(draw) -> pd.DataFrame:
    polars_df = draw(happy_path_df)
    return nw.from_native(polars_df).to_pandas()


@given(happy_path_df | happy_path_ldf | _arrow_strat() | _pandas_strat())
@settings(deadline=None)  # too variant to enforce deadline
def test_datascan_class_parametric(df) -> None:
    scanner = DataScan(data=df)

    df_nw = nw.from_native(df)

    summary_res = scanner.summary_data

    physical_summary = _as_physical(scanner.summary_data)
    physical_input = _as_physical(nw.from_native(df))

    ## High Level Checks:
    cols = summary_res.select("colname").to_dict()["colname"].to_list()

    msg = "cols must be the same"
    df_cols = df_nw.columns
    assert set(cols) == set(df_cols), msg

    msg = "return type is the physical version of the input"
    assert physical_input.implementation == physical_summary.implementation
    assert isinstance(scanner.summary_data, nw.DataFrame)

    msg = "did not return correct amount of summary rows"
    assert len(summary_res) == len(cols)  # only for happy path

    msg = "contains sample data"
    assert "sample_data" in summary_res.columns

    ## More Granular Checks:
    cols_that_must_be_there = ("n_missing", "n_unique", "icon", "colname", "sample_data", "coltype")
    for col in cols_that_must_be_there:
        assert col in summary_res.columns, f"Missing column: {col}"

    # this also catches developer error in syncing the calculations and stat classes
    # for example if dev adds a new stat to `scan_profile_stats.py` and does not add
    # it to the `calc_stats` method, this test will fail since it never calculated the
    # statistic.
    msg = "If a single of a group is there, they should all be there."
    for group in StatGroup:
        stats_that_should_be_present: list[str] = [
            stat.name for stat in COLUMN_ORDER_REGISTRY if group == stat.group
        ]
        any_in_summary = any(
            col for col in stats_that_should_be_present if col in summary_res.columns
        )
        if any_in_summary:
            assert all(stat in summary_res.columns for stat in stats_that_should_be_present), msg


# TODO: Check stats and expressions themselves by determinstic casing


@given(happy_path_df | happy_path_ldf)
def test_datascan_json_output(df):
    scanner = DataScan(data=df)

    profile_json = scanner.to_json()

    assert isinstance(profile_json, str)


@given(df=happy_path_df)
@settings(max_examples=5)
def test_col_summary_tbl(df):
    col_summary = col_summary_tbl(df)

    assert isinstance(col_summary, GT)


def test_col_summary_tbl_polars_categorical_column():
    import polars as pl

    log_levels = pl.Enum(["debug", "info", "warning", "error"])

    df_pl = pl.DataFrame(
        {
            "level": ["debug", "info", "debug", "error"],
            "message": [
                "process id: 525",
                "Service started correctly",
                "startup time: 67ms",
                "Cannot connect to DB!",
            ],
        },
        schema_overrides={
            "level": log_levels,
        },
    )

    tabular_output = col_summary_tbl(df_pl)

    assert isinstance(tabular_output, GT)


def test_datascan_class_raises():
    with pytest.raises(TypeError):
        DataScan(data="not a DataFrame or Ibis Table")

    with pytest.raises(TypeError):
        DataScan(data=123)

    with pytest.raises(TypeError):
        DataScan(data=[1, 2, 3])


def test_compact_integer_fmt():
    assert _compact_integer_fmt(value=0) == "0"
    assert _compact_integer_fmt(value=0.4) == "4.0E−1"
    assert _compact_integer_fmt(value=0.6) == "6.0E−1"
    assert _compact_integer_fmt(value=1) == "1"
    assert _compact_integer_fmt(value=43.91) == "44"
    assert _compact_integer_fmt(value=226.1) == "226"
    assert _compact_integer_fmt(value=4362.54) == "4363"
    assert _compact_integer_fmt(value=15321.23) == "1.5E4"


def test_compact_decimal_fmt():
    assert _compact_decimal_fmt(value=0) == "0.00"
    assert _compact_decimal_fmt(value=1) == "1.00"
    assert _compact_decimal_fmt(value=0.0) == "0.00"
    assert _compact_decimal_fmt(value=1.0) == "1.00"
    assert _compact_decimal_fmt(value=0.1) == "0.10"
    assert _compact_decimal_fmt(value=0.5) == "0.50"
    assert _compact_decimal_fmt(value=0.01) == "0.01"
    assert _compact_decimal_fmt(value=0.009) == "9.0E−3"
    assert _compact_decimal_fmt(value=0.000001) == "1.0E−6"
    assert _compact_decimal_fmt(value=0.99) == "0.99"
    assert _compact_decimal_fmt(value=1) == "1.00"
    assert _compact_decimal_fmt(value=43.91) == "43.9"
    assert _compact_decimal_fmt(value=226.1) == "226"
    assert _compact_decimal_fmt(value=4362.54) == "4363"
    assert _compact_decimal_fmt(value=15321.23) == "1.5E4"


def test_compact_0_1_fmt():
    _compact_0_1_fmt(value=0) == "0.0"
    _compact_0_1_fmt(value=1) == "1.0"
    _compact_0_1_fmt(value=0.0) == "0.0"
    _compact_0_1_fmt(value=1.0) == "1.0"
    _compact_0_1_fmt(value=0.1) == "0.1"
    _compact_0_1_fmt(value=0.5) == "0.5"
    _compact_0_1_fmt(value=0.01) == "0.01"
    _compact_0_1_fmt(value=0.009) == "<0.01"
    _compact_0_1_fmt(value=0.000001) == "<0.01"
    _compact_0_1_fmt(value=0.99) == "0.99"
    _compact_0_1_fmt(value=0.991) == ">0.99"
    _compact_0_1_fmt(value=226.1) == "226"


if __name__ == "__main__":
    pytest.main([__file__, "-x"])
