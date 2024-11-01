from __future__ import annotations

import re

import narwhals as nw
from narwhals.typing import FrameT

def validate_numeric_column(dfn: FrameT, column: str) -> None:
    """
    Validate that the specified column exists in the DataFrame and is numeric.

    Parameters:
    dfn (DataFrame): The DataFrame to check.
    column (str): The column name to validate.

    Raises:
    ValueError: If the column does not exist in the DataFrame.
    TypeError: If the column is not numeric.
    """

    # Expect a Narwhals DataFrame but allow for Pandas/Polars DataFrames
    if not isinstance(dfn, (nw.DataFrame, FrameT)):
        raise TypeError("DataFrame must be a Narwhals, Pandas, or Polars DataFrame.")

    if column not in dfn.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    if not dfn.collect_schema().get(column).is_numeric:
        raise TypeError(f"Column '{column}' is not numeric.")
    def threshold_check(failing_test_units: int, threshold: int) -> bool:
    """
    Determine if the number of failing test units is below the threshold.

    Parameters
    ----------
    failing_test_units
        The number of failing test units.
    threshold
        The maximum number of failing test units to allow.

    Returns
    -------
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    return failing_test_units < threshold
