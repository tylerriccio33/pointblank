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
