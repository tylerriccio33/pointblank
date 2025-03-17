import pytest
import narwhals as nw

from hypothesis import given
import polars.testing.parametric as pt
from great_tables import GT

from pointblank.datascan import DataScan, col_summary_tbl
from pointblank._datascan_utils import _compact_0_1_fmt, _compact_decimal_fmt, _compact_integer_fmt


@given(df=pt.dataframes(min_size=5))
def test_datascan_class_parametric(df) -> None:
    scanner = DataScan(data=df)

    df_nw = nw.from_native(df)

    summary_res = scanner.summary_data

    ## Go through checks:
    cols = summary_res.select("colname").to_polars().to_series().to_list()

    msg = "cols are not the same"
    df_cols = df_nw.columns
    assert set(cols) == set(df_cols), msg


@given(df=pt.dataframes(min_size=5))
def test_datascan_json_output(df):
    scanner = DataScan(data=df)

    profile_json = scanner.to_json()

    assert isinstance(profile_json, str)


@given(df=pt.dataframes(min_size=5))
def test_col_summary_tbl(df):
    col_summary = col_summary_tbl(df)

    assert isinstance(df, GT)


def test_datascan_class_raises():
    with pytest.raises(TypeError):
        DataScan(data="not a DataFrame or Ibis Table")

    with pytest.raises(TypeError):
        DataScan(data=123)

    with pytest.raises(TypeError):
        DataScan(data=[1, 2, 3])


def test_compact_integer_fmt():
    _compact_integer_fmt(value=0) == "0"
    _compact_integer_fmt(value=0.4) == "0"
    _compact_integer_fmt(value=0.6) == "1"
    _compact_integer_fmt(value=1) == "1"
    _compact_integer_fmt(value=43.91) == "44"
    _compact_integer_fmt(value=226.1) == "226"
    _compact_integer_fmt(value=4362.54) == "4363"
    _compact_integer_fmt(value=15321.23) == "15321"


def test_compact_decimal_fmt():
    _compact_decimal_fmt(value=0) == "0.00"
    _compact_decimal_fmt(value=1) == "1.00"
    _compact_decimal_fmt(value=0.0) == "0.00"
    _compact_decimal_fmt(value=1.0) == "1.00"
    _compact_decimal_fmt(value=0.1) == "0.10"
    _compact_decimal_fmt(value=0.5) == "0.50"
    _compact_decimal_fmt(value=0.01) == "0.01"
    _compact_decimal_fmt(value=0.009) == "9.00E-03"
    _compact_decimal_fmt(value=0.000001) == "1.00E-06"
    _compact_decimal_fmt(value=0.99) == "0.99"
    _compact_decimal_fmt(value=1) == "1.00"
    _compact_decimal_fmt(value=43.91) == "43.9"
    _compact_decimal_fmt(value=226.1) == "226"
    _compact_decimal_fmt(value=4362.54) == "4360"
    _compact_decimal_fmt(value=15321.23) == "1.5E4"


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
    pytest.main([__file__])
