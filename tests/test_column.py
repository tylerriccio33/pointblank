import pytest

from pointblank.validate import Validate
from pointblank.column import (
    Column,
    col,
    StartsWith,
    EndsWith,
    Contains,
    Matches,
    Everything,
    FirstN,
    LastN,
    starts_with,
    ends_with,
    contains,
    matches,
    everything,
    first_n,
    last_n,
    AndSelector,
    OrSelector,
    SubSelector,
    NotSelector,
)

import pandas as pd
import polars as pl
import ibis
import narwhals.selectors as ncs


@pytest.fixture
def tbl_pl():
    return pl.DataFrame(
        {
            "a": [1, 2, None, 4, 5, None, None, 8, None],
            "b": [4, None, 6, 7, 8, None, None, None, 12],
            "c": [None, 8, 8, 8, 8, None, 8, None, None],
            "d": [None, 8, 8, 8, 8, None, 8, None, None],
            "e": [9, 9, 9, 9, 9, 9, 9, 9, 9],
            "f": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "g": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def tbl_pd():
    return pd.DataFrame(
        {
            "a": [1, 2, pd.NA, 4, 5, pd.NA, pd.NA, 8, pd.NA],
            "b": [4, pd.NA, 6, 7, 8, pd.NA, pd.NA, pd.NA, 12],
            "c": [pd.NA, 8, 8, 8, 8, pd.NA, 8, pd.NA, pd.NA],
            "d": [pd.NA, 8, 8, 8, 8, pd.NA, 8, pd.NA, pd.NA],
            "e": [9, 9, 9, 9, 9, 9, 9, 9, 9],
            "f": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "g": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def tbl_memtable():
    return ibis.memtable(
        pd.DataFrame(
            {
                "a": [1, 2, pd.NA, 4, 5, pd.NA, pd.NA, 8, pd.NA],
                "b": [4, pd.NA, 6, 7, 8, pd.NA, pd.NA, pd.NA, 12],
                "c": [pd.NA, 8, 8, 8, 8, pd.NA, 8, pd.NA, pd.NA],
                "d": [pd.NA, 8, 8, 8, 8, pd.NA, 8, pd.NA, pd.NA],
                "e": [9, 9, 9, 9, 9, 9, 9, 9, 9],
                "f": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "g": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
    )


@pytest.fixture
def tbl_pl_variable_names():

    return pl.DataFrame(
        {
            "word": ["apple", "banana"],
            "low_numbers": [1, 2],
            "high_numbers": [13500, 95000],
            "low_floats": [41.6, 41.2],
            "high_floats": [41.6, 41.2],
            "superhigh_floats": [23453.23, 32453532.33],
            "date": ["2021-01-01", "2021-01-02"],
            "datetime": ["2021-01-01 00:00:00", "2021-01-02 00:00:00"],
            "bools": [True, False],
        }
    )


@pytest.fixture
def tbl_pd_variable_names():

    return pd.DataFrame(
        {
            "word": ["apple", "banana"],
            "low_numbers": [1, 2],
            "high_numbers": [13500, 95000],
            "low_floats": [41.6, 41.2],
            "high_floats": [41.6, 41.2],
            "superhigh_floats": [23453.23, 32453532.33],
            "date": ["2021-01-01", "2021-01-02"],
            "datetime": ["2021-01-01 00:00:00", "2021-01-02 00:00:00"],
            "bools": [True, False],
        }
    )


@pytest.fixture
def tbl_memtable_variable_names():
    return ibis.memtable(
        pd.DataFrame(
            {
                "word": ["apple", "banana"],
                "low_numbers": [1, 2],
                "high_numbers": [13500, 95000],
                "low_floats": [41.6, 41.2],
                "high_floats": [41.6, 41.2],
                "superhigh_floats": [23453.23, 32453532.33],
                "date": ["2021-01-01", "2021-01-02"],
                "datetime": ["2021-01-01 00:00:00", "2021-01-02 00:00:00"],
                "bools": [True, False],
            }
        )
    )


def test_column_class():
    col1 = Column("col1")
    assert col1.exprs == "col1"
    assert str(col1) == "col1"


def test_col_function():
    col1 = col("col1")
    assert col1.exprs == "col1"
    assert str(col1) == "col1"


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_gt_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_gt(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_gt(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 5
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_lt_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_lt(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_lt(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_eq_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_eq(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="d", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("f"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("f"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="f", value=col("g"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="f", value=col("g"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_ne_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_ne(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("f"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("f"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=7, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=7, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=10, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=10, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=9, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=9, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_ge_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_ge(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="d", value=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="d", value=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_ge(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 5
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_le_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_le(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="b", value=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="b", value=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #    Validate(tbl)
    #    .col_vals_le(columns="e", value=col("d"), na_pass=False)
    #    .interrogate()
    #    .n_passed(i=1, scalar=True)
    #    == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_between_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=0, right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=0, right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=8, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=8, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="c", left=col("b"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="c", left=col("b"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("c"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("c"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #     Validate(tbl)
    #     .col_vals_between(columns="e", left=col("c"), right=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_outside_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=0, right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=0, right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=8, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=8, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="c", left=col("b"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="c", left=col("b"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("c"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("c"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #     Validate(tbl)
    #     .col_vals_between(columns="e", left=col("c"), right=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 0
    # )


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_classes(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # StartsWith tests

    assert Column(exprs=StartsWith("low")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert Column(exprs=StartsWith("LOW")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert Column(exprs=StartsWith("LOW", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert Column(exprs=StartsWith("not_present")).resolve(columns=tbl.columns) == []
    assert Column(exprs=StartsWith("low")).exprs == StartsWith(text="low")

    # EndsWith tests

    assert Column(exprs=EndsWith("floats")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert Column(exprs=EndsWith("FLOATS")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert Column(exprs=EndsWith("FLOATS", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert Column(exprs=EndsWith("not_present")).resolve(columns=tbl.columns) == []
    assert Column(exprs=EndsWith("floats")).exprs == EndsWith(text="floats")

    # Contains tests

    assert Column(exprs=Contains("numbers")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=Contains("NUMBERS")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=Contains("NUMBERS", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert Column(exprs=Contains("not_present")).resolve(columns=tbl.columns) == []
    assert Column(exprs=Contains("numbers")).exprs == Contains(text="numbers")

    # Matches tests

    assert Column(exprs=Matches("at")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert Column(exprs=Matches("numb")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=Matches("NUMB")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=Matches("NUMB", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert Column(exprs=Matches(r"^..._numbers$")).resolve(columns=tbl.columns) == ["low_numbers"]
    assert Column(exprs=Matches("not_present")).resolve(columns=tbl.columns) == []
    assert Column(exprs=Matches("at")).exprs == Matches(pattern="at")

    # Everything tests

    assert Column(exprs=Everything()).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=Everything()).exprs == Everything()

    # FirstN tests

    assert Column(exprs=FirstN(3)).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=FirstN(9)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=FirstN(20)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=FirstN(0)).resolve(columns=tbl.columns) == []
    assert Column(exprs=FirstN(3, offset=1)).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
        "low_floats",
    ]
    assert Column(exprs=FirstN(3, offset=10)).resolve(columns=tbl.columns) == []
    assert Column(exprs=FirstN(0, offset=3)).resolve(columns=tbl.columns) == []
    assert Column(exprs=FirstN(0, offset=10)).resolve(columns=tbl.columns) == []
    assert Column(exprs=FirstN(5, offset=-1)).resolve(columns=tbl.columns) == []

    assert Column(exprs=FirstN(3)).exprs == FirstN(n=3)

    # LastN tests

    assert Column(exprs=LastN(3)).resolve(columns=tbl.columns) == [
        "date",
        "datetime",
        "bools",
    ]
    assert Column(exprs=LastN(9)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=LastN(20)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=LastN(0)).resolve(columns=tbl.columns) == []
    assert Column(exprs=LastN(3, offset=1)).resolve(columns=tbl.columns) == [
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert Column(exprs=LastN(3, offset=10)).resolve(columns=tbl.columns) == []
    assert Column(exprs=LastN(0, offset=3)).resolve(columns=tbl.columns) == []
    assert Column(exprs=LastN(0, offset=10)).resolve(columns=tbl.columns) == []
    assert Column(exprs=LastN(5, offset=-1)).resolve(columns=tbl.columns) == []

    assert Column(exprs=LastN(3)).exprs == LastN(n=3)


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_helper_functions(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert col("not_present").resolve(columns=tbl.columns) == ["not_present"]

    # starts_with() tests

    assert starts_with("low").resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert col(starts_with("low")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert col(starts_with("LOW")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert col(starts_with("LOW", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert col(starts_with("not_present")).resolve(columns=tbl.columns) == []
    assert col(starts_with("low")).exprs == StartsWith(text="low")

    # ends_with() tests

    assert ends_with("floats").resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert col(ends_with("floats")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert col(ends_with("FLOATS")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert col(ends_with("FLOATS", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert col(ends_with("not_present")).resolve(columns=tbl.columns) == []
    assert col(ends_with("floats")).exprs == EndsWith(text="floats")

    # contains() tests

    assert contains("numbers").resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(contains("numbers")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(contains("NUMBERS")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(contains("NUMBERS", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert col(contains("not_present")).resolve(columns=tbl.columns) == []
    assert col(contains("numbers")).exprs == Contains(text="numbers")

    # matches() tests

    assert matches("at").resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert col(matches("at")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert col(matches("numb")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(matches("NUMB")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(matches("NUMB", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert col(matches(r"^..._numbers$")).resolve(columns=tbl.columns) == ["low_numbers"]
    assert col(matches("not_present")).resolve(columns=tbl.columns) == []
    assert col(matches("at")).exprs == Matches(pattern="at")

    # everything() tests

    assert col(everything()).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(everything()).exprs == Everything()

    # first_n() tests

    assert col(first_n(3)).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
    ]
    assert col(first_n(9)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(first_n(20)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(first_n(0)).resolve(columns=tbl.columns) == []
    assert col(first_n(3, offset=1)).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
        "low_floats",
    ]
    assert col(first_n(3, offset=10)).resolve(columns=tbl.columns) == []
    assert col(first_n(0, offset=3)).resolve(columns=tbl.columns) == []
    assert col(first_n(0, offset=10)).resolve(columns=tbl.columns) == []
    assert col(first_n(5, offset=-1)).resolve(columns=tbl.columns) == []

    assert col(first_n(3)).exprs == FirstN(n=3)

    # last_n() tests

    assert col(last_n(3)).resolve(columns=tbl.columns) == [
        "date",
        "datetime",
        "bools",
    ]
    assert col(last_n(9)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(last_n(20)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(last_n(0)).resolve(columns=tbl.columns) == []
    assert col(last_n(3, offset=1)).resolve(columns=tbl.columns) == [
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert col(last_n(3, offset=10)).resolve(columns=tbl.columns) == []
    assert col(last_n(0, offset=3)).resolve(columns=tbl.columns) == []
    assert col(last_n(0, offset=10)).resolve(columns=tbl.columns) == []
    assert col(last_n(5, offset=-1)).resolve(columns=tbl.columns) == []

    assert col(last_n(3)).exprs == LastN(n=3)


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_set_ops_classes(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Column(exprs=AndSelector(StartsWith("low"), EndsWith("floats"))).resolve(
        columns=tbl.columns
    ) == ["low_floats"]

    assert Column(exprs=OrSelector(StartsWith("low"), EndsWith("floats"))).resolve(
        columns=tbl.columns
    ) == ["low_numbers", "low_floats", "high_floats", "superhigh_floats"]

    assert Column(exprs=SubSelector(StartsWith("low"), EndsWith("floats"))).resolve(
        columns=tbl.columns
    ) == ["low_numbers"]

    assert Column(exprs=NotSelector(StartsWith("low"))).resolve(columns=tbl.columns) == [
        "word",
        "high_numbers",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert Column(exprs=NotSelector(EndsWith("floats"))).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "date",
        "datetime",
        "bools",
    ]


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_set_ops_functions(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert col(starts_with("low") & ends_with("floats")).resolve(columns=tbl.columns) == [
        "low_floats"
    ]

    assert col(starts_with("low") | ends_with("floats")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]

    assert col(starts_with("low") - ends_with("floats")).resolve(columns=tbl.columns) == [
        "low_numbers"
    ]

    assert col(~starts_with("low")).resolve(columns=tbl.columns) == [
        "word",
        "high_numbers",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~ends_with("floats")).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~contains("numbers")).resolve(columns=tbl.columns) == [
        "word",
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~matches("at")).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "bools",
    ]

    assert col(~everything()).resolve(columns=tbl.columns) == []

    assert col(~first_n(3)).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~first_n(3, offset=1)).resolve(columns=tbl.columns) == [
        "word",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~last_n(3)).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]

    assert col(~last_n(3, offset=1)).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "bools",
    ]


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_set_ops_functions_complex(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert col(starts_with("low") & ends_with("floats") | contains("numbers")).resolve(
        columns=tbl.columns
    ) == ["low_numbers", "high_numbers", "low_floats"]

    assert col(
        starts_with("low") & ends_with("floats") | contains("numbers") - matches("numb")
    ).resolve(columns=tbl.columns) == ["low_floats"]

    assert col(ends_with("floats") | starts_with("low") - matches("numb")).resolve(
        columns=tbl.columns
    ) == ["low_floats", "high_floats", "superhigh_floats"]

    assert col((starts_with("low") & ends_with("floats")) | contains("numbers")).resolve(
        columns=tbl.columns
    ) == ["low_numbers", "high_numbers", "low_floats"]

    assert col(everything() - (starts_with("low") & ends_with("floats"))).resolve(
        columns=tbl.columns
    ) == [
        "word",
        "low_numbers",
        "high_numbers",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(everything() - starts_with("low") & ends_with("floats")).resolve(
        columns=tbl.columns
    ) == [
        "high_floats",
        "superhigh_floats",
    ]


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names"])
def test_nw_selectors(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Test with single Narwhals column selector
    validation = Validate(data=tbl).col_exists(columns=ncs.numeric()).interrogate()

    assert len(validation.n()) == 5

    # Test with single Narwhals column selector within `col()`
    validation = Validate(data=tbl).col_exists(columns=col(ncs.numeric())).interrogate()

    assert len(validation.n()) == 5

    # Test that multiple Narwhals column selectors work within `col()`
    validation = (
        Validate(data=tbl).col_exists(columns=col(ncs.numeric() | ncs.boolean())).interrogate()
    )

    assert len(validation.n()) == 6
