from __future__ import annotations

from pointblank.utils import (
    column_test_prep,
    threshold_check,
)

from narwhals.typing import FrameT
from pointblank.comparison import Comparator

COL_VALS_COMPARE_ONE_DOCSTRING = """
    Determine if values in a column are ___ a single value.

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
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

COL_VALS_COMPARE_TWO_DOCSTRING = """
    Determine if values in a column are ___ two values.

    Parameters
    ----------
    df : FrameT
        a DataFrame.
    column : str
        The column to check.
    left : float | int
        A lower value to check against.
    right : float | int
        A higher value to check against.
    threshold : int
        The maximum number of failing test units to allow.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

COL_VALS_COMPARE_SET_DOCSTRING = """
    Determine if values in a column are ___ a set of values.

    Parameters
    ----------
    df : FrameT
        a DataFrame.
    column : str
        The column to check.
    values: list[float | int]
        A list of values to check against.
    threshold : int
        The maximum number of failing test units to allow.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """


def _col_vals_compare_one_docstring(comparison: str) -> str:
    """
    Generate a docstring for a column value comparison method.

    Parameters
    ----------
    comparison : str
        The type of comparison ('gt' for greater than, 'lt' for less than).

    Returns
    -------
    str
        The generated docstring.
    """

    return COL_VALS_COMPARE_ONE_DOCSTRING.replace("___", comparison)


def _col_vals_compare_two_docstring(comparison: str) -> str:
    """
    Generate a docstring for a column value comparison method.

    Parameters
    ----------
    comparison : str
        The type of comparison ('between' for between two values).

    Returns
    -------
    str
        The generated docstring.
    """

    return COL_VALS_COMPARE_TWO_DOCSTRING.replace("___", comparison)


def _col_vals_compare_set_docstring(inside: bool) -> str:
    """
    Generate a docstring for a column value comparison method.

    Parameters
    ----------
    inside : bool
        Whether the values should be inside the set.

    Returns
    -------
    str
        The generated docstring.
    """

    comparison = "in" if inside else "not in"
    return COL_VALS_COMPARE_SET_DOCSTRING.replace("___", comparison)


def _col_vals_compare_one(
    df: FrameT,
    column: str,
    value: float | int,
    threshold: int,
    comparison: str,
    type: str = "numeric",
) -> bool:
    """
    General routine to compare values in a column against a single value.

    Parameters
    ----------
    df : FrameT
        a DataFrame.
    column : str
        The column to check.
    value : float | int
        A value to check against.
    threshold : int
        The maximum number of failing test units to allow.
    comparison : str
        The type of comparison ('gt' for greater than, 'lt' for less than).
    type : str
        The data type of the column.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    # Convert the DataFrame to a format that narwhals can work with and:
    #  - check if the column exists
    #  - check if the column type is compatible with the test
    dfn = column_test_prep(df=df, column=column, type=type)

    # Collect results for the test units; the results are a list of booleans where
    # `True` indicates a passing test unit
    if comparison == "gt":
        test_unit_res = Comparator(x=dfn, column=column, compare=value).gt()
    elif comparison == "lt":
        test_unit_res = Comparator(x=dfn, column=column, compare=value).lt()
    elif comparison == "eq":
        test_unit_res = Comparator(x=dfn, column=column, compare=value).eq()
    elif comparison == "ne":
        test_unit_res = Comparator(x=dfn, column=column, compare=value).ne()
    elif comparison == "ge":
        test_unit_res = Comparator(x=dfn, column=column, compare=value).ge()
    elif comparison == "le":
        test_unit_res = Comparator(x=dfn, column=column, compare=value).le()

    else:
        raise ValueError(
            """Invalid comparison type. Use:
            - `gt` for greater than,
            - `lt` for less than,
            - `eq` for equal to,
            - `ne` for not equal to,
            - `ge` for greater than or equal to, or
            - `le` for less than or equal to."""
        )

    # Get the number of failing test units by counting instances of `False` and
    # then determine if the test passes overall by comparing the number of failing
    # test units to the threshold for failing test units
    return threshold_check(failing_test_units=test_unit_res.count(False), threshold=threshold)

