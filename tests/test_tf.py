import pytest
import pandas as pd
import polars as pl

from pointblank.tf import TF


@pytest.fixture
def tbl_pd():
    return pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8], "s": ["a", "b", "c", "d"]}
    )


@pytest.fixture
def tbl_pl():
    return pl.DataFrame(
        {"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8], "s": ["a", "b", "c", "d"]}
    )


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_gt(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert TF.col_vals_gt(tbl, column="x", value=0, threshold=1)
    assert not TF.col_vals_gt(tbl, column="x", value=1, threshold=1)
    assert TF.col_vals_gt(tbl, column="x", value=1, threshold=2)

    assert TF.col_vals_gt(tbl, column="y", value=1, threshold=1)
    assert not TF.col_vals_gt(tbl, column="y", value=4, threshold=1)
    assert TF.col_vals_gt(tbl, column="y", value=4, threshold=2)

    assert not TF.col_vals_gt(tbl, column="z", value=8, threshold=1)
    assert TF.col_vals_gt(tbl, column="z", value=8, threshold=5)

    with pytest.raises(ValueError):
        TF.col_vals_gt(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_lt(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert TF.col_vals_lt(tbl, column="x", value=5, threshold=1)
    assert not TF.col_vals_lt(tbl, column="x", value=4, threshold=1)
    assert TF.col_vals_lt(tbl, column="x", value=4, threshold=2)

    assert TF.col_vals_lt(tbl, column="y", value=8, threshold=1)
    assert not TF.col_vals_lt(tbl, column="y", value=7, threshold=1)
    assert TF.col_vals_lt(tbl, column="y", value=7, threshold=2)

    with pytest.raises(ValueError):
        TF.col_vals_lt(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_eq(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert not TF.col_vals_eq(tbl, column="x", value=8, threshold=1)
    assert TF.col_vals_eq(tbl, column="x", value=8, threshold=5)

    assert not TF.col_vals_eq(tbl, column="y", value=8, threshold=1)
    assert TF.col_vals_eq(tbl, column="y", value=8, threshold=5)

    assert TF.col_vals_eq(tbl, column="z", value=8, threshold=1)
    assert not TF.col_vals_eq(tbl, column="z", value=9, threshold=2)

    with pytest.raises(ValueError):
        TF.col_vals_eq(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_ne(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert TF.col_vals_ne(tbl, column="x", value=5, threshold=1)
    assert not TF.col_vals_ne(tbl, column="x", value=4, threshold=1)
    assert TF.col_vals_ne(tbl, column="x", value=4, threshold=3)

    with pytest.raises(ValueError):
        TF.col_vals_ne(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_ge(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert TF.col_vals_ge(tbl, column="x", value=0, threshold=1)
    assert TF.col_vals_ge(tbl, column="x", value=1, threshold=1)
    assert TF.col_vals_ge(tbl, column="x", value=2, threshold=2)

    assert TF.col_vals_ge(tbl, column="z", value=8, threshold=1)
    assert TF.col_vals_ge(tbl, column="z", value=7, threshold=1)
    assert not TF.col_vals_ge(tbl, column="z", value=9, threshold=1)
    assert TF.col_vals_ge(tbl, column="z", value=9, threshold=5)

    with pytest.raises(ValueError):
        TF.col_vals_ge(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_le(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert TF.col_vals_le(tbl, column="x", value=5, threshold=1)
    assert TF.col_vals_le(tbl, column="x", value=4, threshold=1)
    assert TF.col_vals_le(tbl, column="x", value=3, threshold=2)

    assert TF.col_vals_le(tbl, column="z", value=8, threshold=1)
    assert TF.col_vals_le(tbl, column="z", value=9, threshold=1)
    assert not TF.col_vals_le(tbl, column="z", value=7, threshold=1)
    assert TF.col_vals_le(tbl, column="z", value=7, threshold=5)

    with pytest.raises(ValueError):
        TF.col_vals_le(tbl, column="a", value=0, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_between(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert TF.col_vals_between(tbl, column="x", left=0, right=5, threshold=1)
    assert TF.col_vals_between(tbl, column="x", left=1, right=4, threshold=1)
    assert not TF.col_vals_between(
        tbl, column="x", left=1, right=4, inclusive=(False, False), threshold=1
    )
    assert TF.col_vals_between(tbl, column="x", left=2, right=3, threshold=3)
    assert not TF.col_vals_between(
        tbl, column="x", left=1, right=4, inclusive=(True, False), threshold=1
    )
    assert not TF.col_vals_between(
        tbl, column="x", left=1, right=4, inclusive=(False, True), threshold=1
    )
    assert not TF.col_vals_between(
        tbl, column="x", left=1, right=4, inclusive=(False, False), threshold=2
    )
    with pytest.raises(ValueError):
        TF.col_vals_between(tbl, column="a", left=0, right=5, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_outside(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert not TF.col_vals_outside(tbl, column="x", left=0, right=1)
    assert TF.col_vals_outside(tbl, column="x", left=0, right=1, inclusive=(True, False))
    assert not TF.col_vals_outside(tbl, column="x", left=0, right=1, inclusive=(False, True))
    assert TF.col_vals_outside(tbl, column="x", left=0, right=1, inclusive=(False, False))

    assert TF.col_vals_outside(tbl, column="x", left=5, right=8, threshold=1)
    assert not TF.col_vals_outside(tbl, column="x", left=4, right=8, threshold=1)
    assert TF.col_vals_outside(tbl, column="x", left=4, right=8, threshold=2)
    assert TF.col_vals_outside(
        tbl, column="x", left=4, right=8, inclusive=(False, True), threshold=1
    )
    assert TF.col_vals_outside(
        tbl, column="x", left=4, right=8, inclusive=(False, False), threshold=1
    )

    assert not TF.col_vals_outside(tbl, column="x", left=0, right=10, threshold=4)
    assert TF.col_vals_outside(tbl, column="x", left=0, right=10, threshold=5)

    with pytest.raises(ValueError):
        TF.col_vals_outside(tbl, column="a", left=0, right=5, threshold=1)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_in_set(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert TF.col_vals_in_set(tbl, column="x", set=[1, 2, 3, 4], threshold=1)
    assert not TF.col_vals_in_set(tbl, column="x", set=[2, 3, 4, 5], threshold=1)
    assert TF.col_vals_in_set(tbl, column="x", set=[2, 3, 4, 5], threshold=2)

    assert TF.col_vals_in_set(tbl, column="s", set=["a", "b", "c", "d"], threshold=1)
    assert not TF.col_vals_in_set(tbl, column="s", set=["b", "c", "d", "e"], threshold=1)
    assert TF.col_vals_in_set(tbl, column="s", set=["b", "c", "d", "e"], threshold=2)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_not_in_set(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert TF.col_vals_not_in_set(tbl, column="x", set=[5, 6, 7, 8], threshold=1)
    assert not TF.col_vals_not_in_set(tbl, column="x", set=[4, 5, 6, 7], threshold=1)
    assert TF.col_vals_not_in_set(tbl, column="x", set=[4, 5, 6, 7], threshold=2)

    assert TF.col_vals_not_in_set(tbl, column="s", set=["e", "f", "g", "h"], threshold=1)
    assert not TF.col_vals_not_in_set(tbl, column="s", set=["d", "e", "f", "g"], threshold=1)
    assert TF.col_vals_not_in_set(tbl, column="s", set=["d", "e", "f", "g"], threshold=2)

    assert not TF.col_vals_not_in_set(tbl, column="x", set=[1, 2, 3, 4], threshold=1)
    assert not TF.col_vals_not_in_set(tbl, column="s", set=["a", "b", "c", "d"], threshold=1)
    assert not TF.col_vals_not_in_set(tbl, column="x", set=[1, 2, 3, 4], threshold=1)
    assert not TF.col_vals_not_in_set(tbl, column="s", set=["a", "b", "c", "d"], threshold=1)
