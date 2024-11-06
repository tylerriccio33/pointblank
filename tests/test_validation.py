import pytest
import pandas as pd
import polars as pl

from pointblank.validate import Validate


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


def tbl_missing_pd():
    return pd.DataFrame({"x": [1, 2, pd.NA, 4], "y": [4, pd.NA, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pl():
    return pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, 8, 8, 8]})


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_all_passing(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(column="x", value=0).interrogate()

    assert v.data.shape == (4, 3)
    assert str(v.data["x"].dtype).lower() == "int64"
    assert str(v.data["y"].dtype).lower() == "int64"
    assert str(v.data["z"].dtype).lower() == "int64"

    # There is a single validation check entry in the `validation_info` attribute
    assert len(v.validation_info) == 1

    # The single step had no failing test units so the `all_passed` attribute is `True`
    assert v.all_passed()

    # Test other validation types for all passing behavior in single steps
    assert Validate(tbl).col_vals_lt(column="x", value=5).interrogate().all_passed()
    assert Validate(tbl).col_vals_eq(column="z", value=8).interrogate().all_passed()
    assert Validate(tbl).col_vals_ge(column="x", value=1).interrogate().all_passed()
    assert Validate(tbl).col_vals_le(column="x", value=4).interrogate().all_passed()
    assert Validate(tbl).col_vals_between(column="x", left=0, right=5).interrogate().all_passed()
    assert Validate(tbl).col_vals_outside(column="x", left=-5, right=0).interrogate().all_passed()


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_gt_all_passed(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(column="x", value=0).interrogate()

    # Get the number of passing test units as a dictionary
    n_passed_dict = v.n_passed()
    assert len(n_passed_dict) == 1
    assert n_passed_dict.keys() == {1}
    assert n_passed_dict[1] == 4
