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
def check_column_type(dfn: nw.DataFrame, column: str, type: str) -> None:
    """
    Check if a column is of a certain data type.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
    column
        The column to check for data type.
    dtype
        The data type to check for. The following data types are supported:
        - 'int'
        - 'float'
        - 'str'
        - 'bool'

    Raises
    ------
    TypeError
        When the column is not of the specified data type.
    """

    column_dtype = str(dfn.collect_schema().get(column)).lower()

    if type == "numeric" and not is_numeric_dtype(dtype=column_dtype):
        raise TypeError(f"Column '{column}' is not numeric.")

    if type == "str" and column_dtype != "str":
        raise TypeError(f"Column '{column}' is not a string.")

    if type == "bool" and column_dtype != "bool":
        raise TypeError(f"Column '{column}' is not a boolean.")

    if type == "datetime" and column_dtype != "datetime":
        raise TypeError(f"Column '{column}' is not a datetime.")

    if type == "timedelta" and column_dtype != "timedelta":
        raise TypeError(f"Column '{column}' is not a timedelta.")


def is_numeric_dtype(dtype: str) -> bool:
    """
    Check if a given data type string represents a numeric type.

    Parameters
    ----------
    dtype : str
        The data type string to check.

    Returns
    -------
    bool
        `True` if the data type is numeric, `False` otherwise.
    """
    # Define the regular expression pattern for numeric data types
    numeric_pattern = re.compile(r"^(int|float)\d*$")
    return bool(numeric_pattern.match(dtype))


def column_test_prep(df: FrameT, column: str, type: str) -> nw.DataFrame:

    # Convert the DataFrame to a format that narwhals can work with.
    dfn = convert_to_narwhals(df=df)

    # Check if the column exists
    check_column_exists(dfn=dfn, column=column)

    # Check if the column is numeric. Raise a TypeError if not.
    check_column_type(dfn=dfn, column=column, type=type)

    return dfn


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
