from __future__ import annotations

from pointblank.utils import (
    _column_test_prep,
    _threshold_check,
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
    dfn = _column_test_prep(df=df, column=column, type=type)

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
    return _threshold_check(failing_test_units=test_unit_res.count(False), threshold=threshold)


def _col_vals_compare_two(
    df: FrameT,
    column: str,
    value1: float | int,
    value2: float | int,
    threshold: int,
    comparison: str,
    type: str = "numeric",
) -> bool:
    """
    General routine to compare values in a column against two values.

    Parameters
    ----------
    df : FrameT
        a DataFrame.
    column : str
        The column to check.
    value1 : float | int
        A value to check against.
    value2 : float | int
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
    dfn = _column_test_prep(df=df, column=column, type=type)

    # Collect results for the test units; the results are a list of booleans where
    # `True` indicates a passing test unit
    if comparison == "between":
        test_unit_res = Comparator(x=dfn, column=column, low=value1, high=value2).between()
    elif comparison == "not between":
        test_unit_res = Comparator(x=dfn, column=column, low=value1, high=value2).outside()

    else:
        raise ValueError(
            """Invalid comparison type. Use:
            - `between` for values between two values, or
            - `not between` for values outside two values."""
        )

    # Get the number of failing test units by counting instances of `False` and
    # then determine if the test passes overall by comparing the number of failing
    # test units to the threshold for failing test units
    return _threshold_check(failing_test_units=test_unit_res.count(False), threshold=threshold)


def _col_vals_compare_set(
    df: FrameT,
    column: str,
    values: list[float | int],
    threshold: int,
    inside: bool = True,
    type: str = "numeric",
) -> bool:
    """
    General routine to compare values in a column against a set of values.

    Parameters
    ----------
    df : FrameT
        a DataFrame.
    column : str
        The column to check.
    values : list[float | int]
        A set of values to check against.
    threshold : int
        The maximum number of failing test units to allow.
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
    dfn = _column_test_prep(df=df, column=column, type=type)

    # Collect results for the test units; the results are a list of booleans where
    # `True` indicates a passing test unit
    if inside:
        test_unit_res = Comparator(x=dfn, column=column, compare=values).between()
    else:
        test_unit_res = Comparator(x=dfn, column=column, compare=values).outside()

    # Get the number of failing test units by counting instances of `False` and
    # then determine if the test passes overall by comparing the number of failing
    # test units to the threshold for failing test units
    return _threshold_check(failing_test_units=test_unit_res.count(False), threshold=threshold)


class Test:
    def col_vals_gt(df: FrameT, column: str, value: float | int, threshold: int = 1) -> bool:

        type = "numeric"

        return _col_vals_compare_one(
            df=df, column=column, value=value, threshold=threshold, comparison="gt", type=type
        )

    col_vals_gt.__doc__ = _col_vals_compare_one_docstring(comparison="greater than")

    def col_vals_lt(df: FrameT, column: str, value: float | int, threshold: int = 1) -> bool:

        type = "numeric"

        return _col_vals_compare_one(
            df=df, column=column, value=value, threshold=threshold, comparison="lt", type=type
        )

    col_vals_lt.__doc__ = _col_vals_compare_one_docstring(comparison="less than")

    def col_vals_eq(df: FrameT, column: str, value: float | int, threshold: int = 1) -> bool:

        type = "numeric"

        return _col_vals_compare_one(
            df=df, column=column, value=value, threshold=threshold, comparison="eq", type=type
        )

    col_vals_eq.__doc__ = _col_vals_compare_one_docstring(comparison="equal to")

    def col_vals_ne(df: FrameT, column: str, value: float | int, threshold: int = 1) -> bool:

        type = "numeric"

        return _col_vals_compare_one(
            df=df, column=column, value=value, threshold=threshold, comparison="ne", type=type
        )

    col_vals_ne.__doc__ = _col_vals_compare_one_docstring(comparison="not equal to")

    def col_vals_ge(df: FrameT, column: str, value: float | int, threshold: int = 1) -> bool:

        type = "numeric"

        return _col_vals_compare_one(
            df=df, column=column, value=value, threshold=threshold, comparison="ge", type=type
        )

    col_vals_ge.__doc__ = _col_vals_compare_one_docstring(comparison="greater than or equal to")

    def col_vals_le(df: FrameT, column: str, value: float | int, threshold: int = 1) -> bool:

        type = "numeric"

        return _col_vals_compare_one(
            df=df, column=column, value=value, threshold=threshold, comparison="le", type=type
        )

    col_vals_le.__doc__ = _col_vals_compare_one_docstring(comparison="less than or equal to")

    def col_vals_between(
        df: FrameT, column: str, left: float | int, right: float | int, threshold: int = 1
    ) -> bool:

        type = "numeric"

        return _col_vals_compare_two(
            df=df,
            column=column,
            value1=left,
            value2=right,
            threshold=threshold,
            comparison="between",
            type=type,
        )

    col_vals_between.__doc__ = _col_vals_compare_two_docstring(comparison="between")

    def col_vals_outside(
        df: FrameT, column: str, left: float | int, right: float | int, threshold: int = 1
    ) -> bool:

        type = "numeric"

        return _col_vals_compare_two(
            df=df,
            column=column,
            value1=left,
            value2=right,
            threshold=threshold,
            comparison="outside",
            type=type,
        )

    col_vals_outside.__doc__ = _col_vals_compare_two_docstring(comparison="outside")

    def col_vals_in_set(
        df: FrameT, column: str, values: list[float | int], threshold: int = 1
    ) -> bool:

        type = "numeric"

        return _col_vals_compare_set(
            df=df, column=column, values=values, threshold=threshold, inside=True, type=type
        )

    col_vals_in_set.__doc__ = _col_vals_compare_set_docstring(inside=True)

    def col_vals_not_in_set(
        df: FrameT, column: str, values: list[float | int], threshold: int = 1
    ) -> bool:

        type = "numeric"

        return _col_vals_compare_set(
            df=df, column=column, values=values, threshold=threshold, inside=False, type=type
        )

    col_vals_not_in_set.__doc__ = _col_vals_compare_set_docstring(inside=False)
