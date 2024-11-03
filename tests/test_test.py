import pytest
import pandas as pd
import polars as pl

from pointblank.test import Test


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_gt(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Test.col_vals_gt(tbl, column="x", value=0, threshold=1) == True
    assert Test.col_vals_gt(tbl, column="x", value=1, threshold=1) == False
    assert Test.col_vals_gt(tbl, column="x", value=1, threshold=2) == True

    assert Test.col_vals_gt(tbl, column="y", value=1, threshold=1) == True
    assert Test.col_vals_gt(tbl, column="y", value=4, threshold=1) == False
    assert Test.col_vals_gt(tbl, column="y", value=4, threshold=2) == True

    assert Test.col_vals_gt(tbl, column="z", value=8, threshold=1) == False
    assert Test.col_vals_gt(tbl, column="z", value=8, threshold=5) == True

    with pytest.raises(ValueError):
        Test.col_vals_gt(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_lt(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Test.col_vals_lt(tbl, column="x", value=5, threshold=1) == True
    assert Test.col_vals_lt(tbl, column="x", value=4, threshold=1) == False
    assert Test.col_vals_lt(tbl, column="x", value=4, threshold=2) == True

    assert Test.col_vals_lt(tbl, column="y", value=8, threshold=1) == True
    assert Test.col_vals_lt(tbl, column="y", value=7, threshold=1) == False
    assert Test.col_vals_lt(tbl, column="y", value=7, threshold=2) == True

    with pytest.raises(ValueError):
        Test.col_vals_lt(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_eq(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Test.col_vals_eq(tbl, column="x", value=8, threshold=1) == False
    assert Test.col_vals_eq(tbl, column="x", value=8, threshold=5) == True

    assert Test.col_vals_eq(tbl, column="y", value=8, threshold=1) == False
    assert Test.col_vals_eq(tbl, column="y", value=8, threshold=5) == True

    assert Test.col_vals_eq(tbl, column="z", value=8, threshold=1) == True
    assert Test.col_vals_eq(tbl, column="z", value=9, threshold=2) == False

    with pytest.raises(ValueError):
        Test.col_vals_eq(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_ne(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Test.col_vals_ne(tbl, column="x", value=5, threshold=1) == True
    assert Test.col_vals_ne(tbl, column="x", value=4, threshold=1) == False
    assert Test.col_vals_ne(tbl, column="x", value=4, threshold=3) == True

    with pytest.raises(ValueError):
        Test.col_vals_ne(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_ge(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Test.col_vals_ge(tbl, column="x", value=0, threshold=1) == True
    assert Test.col_vals_ge(tbl, column="x", value=1, threshold=1) == True
    assert Test.col_vals_ge(tbl, column="x", value=2, threshold=2) == True

    assert Test.col_vals_ge(tbl, column="z", value=8, threshold=1) == True
    assert Test.col_vals_ge(tbl, column="z", value=7, threshold=1) == True
    assert Test.col_vals_ge(tbl, column="z", value=9, threshold=1) == False
    assert Test.col_vals_ge(tbl, column="z", value=9, threshold=5) == True

    with pytest.raises(ValueError):
        Test.col_vals_ge(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_le(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Test.col_vals_le(tbl, column="x", value=5, threshold=1) == True
    assert Test.col_vals_le(tbl, column="x", value=4, threshold=1) == True
    assert Test.col_vals_le(tbl, column="x", value=3, threshold=2) == True

    assert Test.col_vals_le(tbl, column="z", value=8, threshold=1) == True
    assert Test.col_vals_le(tbl, column="z", value=9, threshold=1) == True
    assert Test.col_vals_le(tbl, column="z", value=7, threshold=1) == False
    assert Test.col_vals_le(tbl, column="z", value=7, threshold=5) == True

    with pytest.raises(ValueError):
        Test.col_vals_le(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_between(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Test.col_vals_between(tbl, column="x", left=0, right=5, threshold=1) == True
    assert Test.col_vals_between(tbl, column="x", left=1, right=5, threshold=1) == False
    assert Test.col_vals_between(tbl, column="x", left=0, right=4, threshold=1) == False
    assert Test.col_vals_between(tbl, column="x", left=1, right=4, threshold=2) == False
    assert Test.col_vals_between(tbl, column="x", left=1, right=4, threshold=3) == True

    with pytest.raises(ValueError):
        Test.col_vals_between(tbl, column="a", left=0, right=5, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_outside(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Test.col_vals_outside(tbl, column="x", left=5, right=8, threshold=1) == True
    assert Test.col_vals_outside(tbl, column="x", left=4, right=8, threshold=1) == False
    assert Test.col_vals_outside(tbl, column="x", left=4, right=8, threshold=2) == True
    assert Test.col_vals_outside(tbl, column="x", left=0, right=6, threshold=1) == False
    assert Test.col_vals_outside(tbl, column="x", left=0, right=6, threshold=5) == True

    with pytest.raises(ValueError):
        Test.col_vals_outside(tbl, column="a", left=0, right=5, threshold=1)
