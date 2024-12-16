import pytest

import pathlib

from pointblank.validate import Validate
from pointblank.column import Column, col

import pandas as pd
import polars as pl
import ibis


@pytest.fixture
def tbl_missing_pd():
    return pd.DataFrame({"x": [1, 2, pd.NA, 4], "y": [4, pd.NA, 6, 7], "z": [8, pd.NA, 8, 8]})


@pytest.fixture
def tbl_missing_pl():
    return pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, None, 8, 8]})


@pytest.fixture
def tbl_missing_ibis_memtable():
    return ibis.memtable(
        pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, None, 8, 8]})
    )


def test_column_class():
    col1 = Column(name="col1")
    assert col1.name == "col1"
    assert str(col1) == "col1"


def test_col_function():
    col1 = col("col1")
    assert col1.name == "col1"
    assert str(col1) == "col1"


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_missing_pd", "tbl_missing_pl", "tbl_missing_ibis_memtable"]
)
def test_col_vals_gt_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl.head(2))
        .col_vals_gt(columns="y", value=col("x"))
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )
    assert (
        Validate(tbl.head(2))
        .col_vals_gt(columns="y", value=col("x"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_missing_pd", "tbl_missing_pl"])
def test_col_vals_lt_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl.head(2))
        .col_vals_lt(columns="x", value=col("y"))
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )
    assert (
        Validate(tbl.head(2))
        .col_vals_lt(columns="x", value=col("y"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_missing_pd", "tbl_missing_pl"])
def test_col_vals_eq_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    if tbl_fixture == "tbl_missing_pd":
        # Add the zz column which is a near duplicate of the z column
        tbl_eq = tbl.assign(zz=[pd.NA, 8, 8, 8])
    else:
        tbl_eq = tbl.with_columns(pl.Series(name="zz", values=[None, 8, 8, 8]))

    assert (
        Validate(tbl_eq).col_vals_eq(columns="zz", value=col("z")).interrogate().n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl_eq).col_vals_eq(columns="z", value=col("zz")).interrogate().n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl_eq)
        .col_vals_eq(columns="zz", value=col("z"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl_eq)
        .col_vals_eq(columns="z", value=col("zz"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_missing_pd", "tbl_missing_pl"])
def test_col_vals_ne_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl).col_vals_ne(columns="x", value=col("y")).interrogate().n_passed(i=1)[1] == 2
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="x", value=col("y"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_missing_pd", "tbl_missing_pl"])
def test_col_vals_ge_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl).col_vals_ge(columns="z", value=col("x")).interrogate().n_passed(i=1)[1] == 2
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="z", value=col("x"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_missing_pd", "tbl_missing_pl"])
def test_col_vals_le_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl).col_vals_le(columns="x", value=col("y")).interrogate().n_passed(i=1)[1] == 2
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="x", value=col("y"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_missing_pd", "tbl_missing_pl"])
def test_col_vals_between_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_between(columns="y", left=col("x"), right=col("z"))
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="y", left=col("x"), right=col("z"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_missing_pd", "tbl_missing_pl"])
def test_col_vals_outside_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_outside(columns="z", left=col("x"), right=col("y"))
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="z", left=col("x"), right=col("y"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
