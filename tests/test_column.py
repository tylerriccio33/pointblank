import pytest

from pointblank.validate import Validate
from pointblank.column import Column, col

import pandas as pd
import polars as pl
import ibis


@pytest.fixture
def tbl_pl():
    return pl.DataFrame(
        {
            "a": [1, 2, None, 4, 5, None, None, 8, None],
            "b": [4, None, 6, 7, 8, None, None, None, 12],
            "c": [None, 8, 8, 8, 8, None, 8, None, None],
            "d": [None, 8, 8, 8, 8, None, 8, None, None],
            "e": [9, 9, 9, 9, 9, 9, 9, 9, 9],
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
            }
        )
    )


def test_column_class():
    col1 = Column(name="col1")
    assert col1.name == "col1"
    assert str(col1) == "col1"


def test_col_function():
    col1 = col("col1")
    assert col1.name == "col1"
    assert str(col1) == "col1"


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_gt_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_gt(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_gt(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1)[1]
    #     == 5
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_lt_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_lt(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_lt(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1)[1]
    #     == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_eq_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_eq(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="d", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_ne_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_ne(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_ge_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_ge(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="d", value=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="d", value=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_ge(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1)[1]
    #     == 5
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_le_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_le(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="b", value=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="b", value=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 5
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #    Validate(tbl)
    #    .col_vals_le(columns="e", value=col("d"), na_pass=False)
    #    .interrogate()
    #    .n_passed(i=1)[1]
    #    == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_between_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=0, right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=0, right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=8, na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=8, na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="c", left=col("b"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="c", left=col("b"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("c"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("c"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 6
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #     Validate(tbl)
    #     .col_vals_between(columns="e", left=col("c"), right=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1)[1]
    #     == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_outside_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=0, right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=0, right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=8, na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=8, na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="c", left=col("b"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="c", left=col("b"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("c"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("c"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1)[1]
        == 9
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #     Validate(tbl)
    #     .col_vals_between(columns="e", left=col("c"), right=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1)[1]
    #     == 0
    # )
