import pytest
import pandas as pd
import polars as pl

import sys
from unittest.mock import patch

import narwhals as nw

from pointblank._utils import (
    _convert_to_narwhals,
    _check_column_exists,
    _is_numeric_dtype,
    _is_date_or_datetime_dtype,
    _is_duration_dtype,
    _get_column_dtype,
    _check_column_type,
    _column_test_prep,
    _get_fn_name,
    _get_assertion_from_fname,
    _check_invalid_fields,
    _select_df_lib,
    _get_tbl_type,
)


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pd():
    return pd.DataFrame({"x": [1, 2, pd.NA, 4], "y": [4, pd.NA, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pl():
    return pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_multiple_types_pd():
    return pd.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float": [4.0, 5.0, 6.0, 7.0],
            "str": ["a", "b", "c", "d"],
            "bool": [True, False, True, False],
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]),
            "datetime": pd.to_datetime(
                [
                    "2021-01-01 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-03 00:00:00",
                    "2021-01-04 00:00:00",
                ]
            ),
            "timedelta": pd.to_timedelta(["1 days", "2 days", "3 days", "4 days"]),
        }
    )


@pytest.fixture
def tbl_multiple_types_pl():
    # Create a Polars DataFrame with multiple data types (int, float, str, date, datetime, timedelta)
    return pl.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float": [4.0, 5.0, 6.0, 7.0],
            "str": ["a", "b", "c", "d"],
            "bool": [True, False, True, False],
            "date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "datetime": [
                "2021-01-01 00:00:00",
                "2021-01-02 00:00:00",
                "2021-01-03 00:00:00",
                "2021-01-04 00:00:00",
            ],
            "timedelta": [1, 2, 3, 4],
        }
    ).with_columns(
        date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        datetime=pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        timedelta=pl.duration(days=pl.col("timedelta")),
    )


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_convert_to_narwhals(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    assert isinstance(dfn, nw.DataFrame)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_double_convert_to_narwhals(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)
    dfn_2 = _convert_to_narwhals(dfn)

    assert isinstance(dfn_2, nw.DataFrame)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_check_column_exists_no_error(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    _check_column_exists(dfn=dfn, column="x")
    _check_column_exists(dfn=dfn, column="y")
    _check_column_exists(dfn=dfn, column="z")


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_missing_pd", "tbl_missing_pl"],
)
def test_check_column_exists_missing_values_no_error(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    _check_column_exists(dfn=dfn, column="x")
    _check_column_exists(dfn=dfn, column="y")
    _check_column_exists(dfn=dfn, column="z")


def test_is_numeric_dtype():

    assert _is_numeric_dtype(dtype="int")
    assert _is_numeric_dtype(dtype="float")
    assert _is_numeric_dtype(dtype="int64")
    assert _is_numeric_dtype(dtype="float64")


def test_is_date_or_datetime_dtype():

    assert _is_date_or_datetime_dtype(dtype="datetime")
    assert _is_date_or_datetime_dtype(dtype="date")
    assert _is_date_or_datetime_dtype(dtype="datetime(time_unit='ns', time_zone=none)")
    assert _is_date_or_datetime_dtype(dtype="datetime(time_unit='us', time_zone=none)")


def test_is_duration_dtype():

    assert _is_duration_dtype(dtype="duration")
    assert _is_duration_dtype(dtype="duration(time_unit='ns')")
    assert _is_duration_dtype(dtype="duration(time_unit='us')")


def test_get_column_dtype_pd(tbl_multiple_types_pd):

    dfn = _convert_to_narwhals(tbl_multiple_types_pd)

    assert _get_column_dtype(dfn=dfn, column="int") == "int64"
    assert _get_column_dtype(dfn=dfn, column="float") == "float64"
    assert _get_column_dtype(dfn=dfn, column="str") == "string"
    assert _get_column_dtype(dfn=dfn, column="bool") == "boolean"
    assert _get_column_dtype(dfn=dfn, column="date") == "datetime(time_unit='ns', time_zone=none)"
    assert (
        _get_column_dtype(dfn=dfn, column="datetime") == "datetime(time_unit='ns', time_zone=none)"
    )
    assert _get_column_dtype(dfn=dfn, column="timedelta") == "duration(time_unit='ns')"

    assert _get_column_dtype(dfn=dfn, column="int", lowercased=False) == "Int64"
    assert _get_column_dtype(dfn=dfn, column="float", lowercased=False) == "Float64"
    assert _get_column_dtype(dfn=dfn, column="str", lowercased=False) == "String"


def test_get_column_dtype_pl(tbl_multiple_types_pl):

    dfn = _convert_to_narwhals(tbl_multiple_types_pl)

    assert _get_column_dtype(dfn=dfn, column="int") == "int64"
    assert _get_column_dtype(dfn=dfn, column="float") == "float64"
    assert _get_column_dtype(dfn=dfn, column="str") == "string"
    assert _get_column_dtype(dfn=dfn, column="bool") == "boolean"
    assert _get_column_dtype(dfn=dfn, column="date") == "date"
    assert (
        _get_column_dtype(dfn=dfn, column="datetime") == "datetime(time_unit='us', time_zone=none)"
    )
    assert _get_column_dtype(dfn=dfn, column="timedelta") == "duration(time_unit='us')"

    assert _get_column_dtype(dfn=dfn, column="int", lowercased=False) == "Int64"
    assert _get_column_dtype(dfn=dfn, column="float", lowercased=False) == "Float64"
    assert _get_column_dtype(dfn=dfn, column="str", lowercased=False) == "String"


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_type(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    _check_column_type(dfn=dfn, column="int", allowed_types=["numeric"])
    _check_column_type(dfn=dfn, column="int", allowed_types=["numeric", "str"])
    _check_column_type(dfn=dfn, column="int", allowed_types=["numeric", "str", "bool"])

    _check_column_type(dfn=dfn, column="float", allowed_types=["numeric"])
    _check_column_type(dfn=dfn, column="float", allowed_types=["numeric", "str"])
    _check_column_type(dfn=dfn, column="float", allowed_types=["numeric", "str", "bool"])

    _check_column_type(dfn=dfn, column="str", allowed_types=["str"])
    _check_column_type(dfn=dfn, column="str", allowed_types=["str", "numeric"])
    _check_column_type(dfn=dfn, column="str", allowed_types=["str", "numeric", "bool"])

    _check_column_type(dfn=dfn, column="bool", allowed_types=["bool"])
    _check_column_type(dfn=dfn, column="bool", allowed_types=["bool", "numeric"])
    _check_column_type(dfn=dfn, column="bool", allowed_types=["bool", "numeric", "str"])

    _check_column_type(dfn=dfn, column="datetime", allowed_types=["datetime"])
    _check_column_type(dfn=dfn, column="datetime", allowed_types=["datetime", "str"])
    _check_column_type(dfn=dfn, column="datetime", allowed_types=["datetime", "str", "numeric"])

    _check_column_type(dfn=dfn, column="timedelta", allowed_types=["duration"])
    _check_column_type(dfn=dfn, column="timedelta", allowed_types=["duration", "str"])
    _check_column_type(dfn=dfn, column="timedelta", allowed_types=["duration", "str", "numeric"])


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_type_raises(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="int", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="float", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="str", allowed_types=["numeric"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="bool", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="date", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="datetime", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="timedelta", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="int", allowed_types=["bool"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="float", allowed_types=["bool"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="str", allowed_types=["bool"])


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_type_raises_invalid_input(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    with pytest.raises(ValueError):
        _check_column_type(dfn=dfn, column="int", allowed_types=[])

    with pytest.raises(ValueError):
        _check_column_type(
            dfn=dfn, column="int", allowed_types=["numeric", "str", "bool", "invalid"]
        )


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_test_prep(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    _column_test_prep(df=tbl, column="int", allowed_types=["numeric"])
    _column_test_prep(df=tbl, column="float", allowed_types=["numeric"])
    _column_test_prep(df=tbl, column="str", allowed_types=["str"])
    _column_test_prep(df=tbl, column="bool", allowed_types=["bool"])
    _column_test_prep(df=tbl, column="date", allowed_types=["datetime"])
    _column_test_prep(df=tbl, column="datetime", allowed_types=["datetime"])
    _column_test_prep(df=tbl, column="timedelta", allowed_types=["duration"])

    # Using `allowed_types=None` bypasses the type check
    _column_test_prep(df=tbl, column="int", allowed_types=None)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_test_prep_raises(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    # No types in `allowed_types` match the column data type
    with pytest.raises(TypeError):
        _column_test_prep(
            df=tbl, column="int", allowed_types=["str", "bool", "datetime", "duration"]
        )

    # Column not present in DataFrame
    with pytest.raises(ValueError):
        _column_test_prep(df=tbl, column="invalid", allowed_types=["numeric"])


def test_get_fn_name():

    def get_name():
        return _get_fn_name()

    assert get_name() == "get_name"


def test_get_assertion_from_fname():

    def col_vals_gt():
        return _get_assertion_from_fname()

    def col_vals_lt():
        return _get_assertion_from_fname()

    def col_vals_eq():
        return _get_assertion_from_fname()

    def col_vals_ne():
        return _get_assertion_from_fname()

    def col_vals_ge():
        return _get_assertion_from_fname()

    def col_vals_le():
        return _get_assertion_from_fname()

    def col_vals_between():
        return _get_assertion_from_fname()

    def col_vals_outside():
        return _get_assertion_from_fname()

    def col_vals_in_set():
        return _get_assertion_from_fname()

    def col_vals_not_in_set():
        return _get_assertion_from_fname()

    assert col_vals_gt() == "gt"
    assert col_vals_lt() == "lt"
    assert col_vals_eq() == "eq"
    assert col_vals_ne() == "ne"
    assert col_vals_ge() == "ge"
    assert col_vals_le() == "le"
    assert col_vals_between() == "between"
    assert col_vals_outside() == "outside"
    assert col_vals_in_set() == "in_set"
    assert col_vals_not_in_set() == "not_in_set"


def test_check_invalid_fields():

    with pytest.raises(ValueError):
        _check_invalid_fields(
            fields=["invalid"], valid_fields=["numeric", "str", "bool", "datetime", "duration"]
        )

    with pytest.raises(ValueError):
        _check_invalid_fields(
            fields=["numeric", "str", "bool", "datetime", "duration", "invalid"],
            valid_fields=["numeric", "str", "bool", "datetime", "duration"],
        )


def test_select_df_lib():

    # Mock the absence of the both the Pandas and Polars libraries
    with patch.dict(sys.modules, {"pandas": None, "polars": None}):
        # An ImportError is raised when the `pandas` and `polars` packages are not installed
        with pytest.raises(ImportError):
            _select_df_lib()

    # Mock the absence of the Pandas library
    with patch.dict(sys.modules, {"pandas": None}):
        # The Polars library is selected when the `pandas` package is not installed
        assert _select_df_lib(preference="polars") == pl
        assert _select_df_lib(preference="pandas") == pl

    # Mock the absence of the Polars library
    with patch.dict(sys.modules, {"polars": None}):
        # The Pandas library is selected when the `polars` package is not installed
        assert _select_df_lib(preference="pandas") == pd
        assert _select_df_lib(preference="polars") == pd

    # Where both the Pandas and Polars libraries are available
    assert _select_df_lib(preference="pandas") == pd
    assert _select_df_lib(preference="polars") == pl


def test_get_tbl_type():

    assert _get_tbl_type(pd.DataFrame()) == "pandas"
    assert _get_tbl_type(pl.DataFrame()) == "polars"
