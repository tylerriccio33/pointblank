import pytest
import pandas as pd

from pointblank.test import Test


@pytest.fixture
def tbl():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})

