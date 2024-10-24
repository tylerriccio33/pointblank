from __future__ import annotations

import narwhals as nw
from narwhals.typing import FrameT

def validate_numeric_column(dfn: FrameT, column):
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
    