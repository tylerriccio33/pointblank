from __future__ import annotations

import re

import narwhals as nw
from narwhals.typing import FrameT


def _convert_to_narwhals(df: FrameT) -> nw.DataFrame:

    # Convert the DataFrame to a format that narwhals can work with
    return nw.from_native(df)


def _check_column_exists(dfn: nw.DataFrame, column: str) -> None:
    """
    Check if a column exists in a DataFrame.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
    column
        The column to check for existence.

    Raises
    ------
    ValueError
        When the column is not found in the DataFrame.
    """

    if column not in dfn.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")


def _is_numeric_dtype(dtype: str) -> bool:
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


def _check_column_type(dfn: nw.DataFrame, column: str, type: str) -> None:
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

    if type == "numeric" and not _is_numeric_dtype(dtype=column_dtype):
        raise TypeError(f"Column '{column}' is not numeric.")

    if type == "str" and column_dtype != "str":
        raise TypeError(f"Column '{column}' is not a string.")

    if type == "bool" and column_dtype != "bool":
        raise TypeError(f"Column '{column}' is not a boolean.")

    if type == "datetime" and column_dtype != "datetime":
        raise TypeError(f"Column '{column}' is not a datetime.")

    if type == "timedelta" and column_dtype != "timedelta":
        raise TypeError(f"Column '{column}' is not a timedelta.")


def _column_test_prep(df: FrameT, column: str, type: str) -> nw.DataFrame:  # noqa: F401

    # Convert the DataFrame to a format that narwhals can work with.
    dfn = _convert_to_narwhals(df=df)

    # Check if the column exists
    _check_column_exists(dfn=dfn, column=column)

    # Check if the column is numeric. Raise a TypeError if not.
    _check_column_type(dfn=dfn, column=column, type=type)

    return dfn


def _threshold_check(failing_test_units: int | float, threshold: int | float) -> bool:
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
