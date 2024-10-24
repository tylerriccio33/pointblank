from __future__ import annotations

import narwhals as nw
from narwhals.typing import FrameT
from confirm.comparison import Comparator


def test_col_vals_gt(df: FrameT, column: str, value: float | int, threshold: int = 1) -> bool:
    """
    Determine if values in a column are greater than a single value.

    Parameters
    ----------
    object
        a DataFrame.
    column
        The column to check.
    value
        A value to check against.
    threshold
        The maximum number of failing test units to allow.

    Returns
    -------
        `True` when test units pass below the threshold level for failing test units, `False` otherwise.
    """

    # Convert the DataFrame to a format that narwhals can work with.
    dfn = nw.from_native(df)

    # Check if the column exists.
    if column not in dfn.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # Check if the column is numeric. Raise a TypeError if not.
    if not dfn.collect_schema().get(column).is_numeric:
        raise TypeError(f"Column '{column}' is not numeric.")

    # Get values from the column as a list
    value_list = dfn[column].to_list()

    # Collect results for the test units; the results are a list of booleans where
    # `True` indicates a passing test unit
    test_unit_res = Comparator(value_list, value).gt()

    # Get the number of failing test units by counting instances of False
    failing_test_units = test_unit_res.count(False)

    # Determine if the test passes overall by comparing the number of failing 
    # test units to the threshold for failing test units
    return failing_test_units < threshold
    