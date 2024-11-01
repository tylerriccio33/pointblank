import pytest
import pandas as pd

from pointblank.test import Test


@pytest.fixture
def tbl():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


def test_col_vals_gt(tbl):

    assert Test.col_vals_gt(tbl, column="x", value=0, threshold=1) == True
    assert Test.col_vals_gt(tbl, column="x", value=1, threshold=1) == False
    assert Test.col_vals_gt(tbl, column="x", value=1, threshold=2) == True

    assert Test.col_vals_gt(tbl, column="y", value=1, threshold=1) == True
    assert Test.col_vals_gt(tbl, column="y", value=4, threshold=1) == False
    assert Test.col_vals_gt(tbl, column="y", value=4, threshold=2) == True

    assert Test.col_vals_gt(tbl, column="z", value=8, threshold=1) == False
    assert Test.col_vals_gt(tbl, column="z", value=8, threshold=5) == True

