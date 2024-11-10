from __future__ import annotations

import re
import inspect

import narwhals as nw
from narwhals.typing import FrameT

from pointblank._constants import TYPE_METHOD_MAP, GENERAL_COLUMN_TYPES


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


def _check_column_type(dfn: nw.DataFrame, column: str, allowed_types: list[str]) -> None:
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

    if _is_numeric_dtype(column_dtype) and "numeric" not in allowed_types:
        raise TypeError(f"Column '{column}' is numeric.")

    if column_dtype == "str" and "str" not in allowed_types:
        raise TypeError(f"Column '{column}' is a string.")

    if column_dtype == "bool" and "bool" not in allowed_types:
        raise TypeError(f"Column '{column}' is a boolean.")

    if column_dtype == "datetime" and "datetime" not in allowed_types:
        raise TypeError(f"Column '{column}' is a datetime.")

    if column_dtype == "timedelta" and "timedelta" not in allowed_types:
        raise TypeError(f"Column '{column}' is a timedelta.")


def _column_test_prep(df: FrameT, column: str, allowed_types: list[str]) -> nw.DataFrame:

    # Convert the DataFrame to a format that narwhals can work with.
    dfn = _convert_to_narwhals(df=df)

    # Check if the column exists
    _check_column_exists(dfn=dfn, column=column)

    # Check if the column is numeric. Raise a TypeError if not.
    _check_column_type(dfn=dfn, column=column, allowed_types=allowed_types)

    return dfn


def _get_comparison_from_fname() -> str:

    # Get the current function name
    func_name = inspect.currentframe().f_back.f_code.co_name

    # Use the `TYPE_METHOD_MAP` dictionary to get the comparison type
    comparison = TYPE_METHOD_MAP.get(func_name)

    return comparison


def _get_def_name() -> str:

    # Get the current function name
    assertion_type = inspect.currentframe().f_back.f_code.co_name

    return assertion_type
