from __future__ import annotations

from dataclasses import dataclass

from narwhals.typing import FrameT

from pointblank._constants import COMPATIBLE_TYPES
from pointblank._comparison import (
    ColValsCompareOne,
    ColValsCompareTwo,
    ColValsCompareSet,
)
from pointblank._utils import _get_comparison_from_fname


COL_VALS_COMPARE_ONE_DOCSTRING = """
    Test whether values in a column are ___ a single value.

    Parameters
    ----------
    df : FrameT
        a DataFrame.
    column : str
        The column to check.
    value : float | int
        A value to check against.
    na_pass : bool
        Whether to pass rows with missing values.
    threshold : int
        The maximum number of failing test units to allow.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

COL_VALS_COMPARE_TWO_DOCSTRING = """
    Test whether values in a column are ___ two values.

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
    inclusive : tuple[bool, bool]
        A tuple of two boolean values indicating whether the comparison should be inclusive. The
        position of the boolean values correspond to the `left=` and `right=` values, respectively.
    na_pass : bool
        Whether to pass rows with missing values.
    threshold : int
        The maximum number of failing test units to allow.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

COL_VALS_COMPARE_SET_DOCSTRING = """
    Test whether values in a column are ___ a set of values.

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


@dataclass
class TF:
    """
    Tests use tabular data and return a single boolean value per check.
    """

    def col_vals_gt(
        df: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        comparison = _get_comparison_from_fname()
        compatible_types = COMPATIBLE_TYPES.get(comparison, [])

        return ColValsCompareOne(
            df=df,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            comparison=comparison,
            allowed_types=compatible_types,
        ).test()

    col_vals_gt.__doc__ = _col_vals_compare_one_docstring(comparison="greater than")

    def col_vals_lt(
        df: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        comparison = _get_comparison_from_fname()
        compatible_types = COMPATIBLE_TYPES.get(comparison, [])

        return ColValsCompareOne(
            df=df,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            comparison=comparison,
            allowed_types=compatible_types,
        ).test()

    col_vals_lt.__doc__ = _col_vals_compare_one_docstring(comparison="less than")

    def col_vals_eq(
        df: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        comparison = _get_comparison_from_fname()
        compatible_types = COMPATIBLE_TYPES.get(comparison, [])

        return ColValsCompareOne(
            df=df,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            comparison=comparison,
            allowed_types=compatible_types,
        ).test()

    col_vals_eq.__doc__ = _col_vals_compare_one_docstring(comparison="equal to")

    def col_vals_ne(
        df: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        comparison = _get_comparison_from_fname()
        compatible_types = COMPATIBLE_TYPES.get(comparison, [])

        return ColValsCompareOne(
            df=df,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            comparison=comparison,
            allowed_types=compatible_types,
        ).test()

    col_vals_ne.__doc__ = _col_vals_compare_one_docstring(comparison="not equal to")

    def col_vals_ge(
        df: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        comparison = _get_comparison_from_fname()
        compatible_types = COMPATIBLE_TYPES.get(comparison, [])

        return ColValsCompareOne(
            df=df,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            comparison=comparison,
            allowed_types=compatible_types,
        ).test()

    col_vals_ge.__doc__ = _col_vals_compare_one_docstring(comparison="greater than or equal to")

    def col_vals_le(
        df: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        comparison = _get_comparison_from_fname()
        compatible_types = COMPATIBLE_TYPES.get(comparison, [])

        return ColValsCompareOne(
            df=df,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            comparison=comparison,
            allowed_types=compatible_types,
        ).test()

    col_vals_le.__doc__ = _col_vals_compare_one_docstring(comparison="less than or equal to")

    def col_vals_between(
        df: FrameT,
        column: str,
        left: float | int,
        right: float | int,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        threshold: int = 1,
    ) -> bool:

        comparison = _get_comparison_from_fname()
        compatible_types = COMPATIBLE_TYPES.get(comparison, [])

        return ColValsCompareTwo(
            df=df,
            column=column,
            value1=left,
            value2=right,
            inclusive=inclusive,
            na_pass=na_pass,
            threshold=threshold,
            comparison=comparison,
            allowed_types=compatible_types,
        ).test()

    col_vals_between.__doc__ = _col_vals_compare_two_docstring(comparison="between")

    def col_vals_outside(
        df: FrameT,
        column: str,
        left: float | int,
        right: float | int,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        threshold: int = 1,
    ) -> bool:

        comparison = _get_comparison_from_fname()
        compatible_types = COMPATIBLE_TYPES.get(comparison, [])

        return ColValsCompareTwo(
            df=df,
            column=column,
            value1=left,
            value2=right,
            inclusive=inclusive,
            na_pass=na_pass,
            threshold=threshold,
            comparison=comparison,
            allowed_types=compatible_types,
        ).test()

    col_vals_outside.__doc__ = _col_vals_compare_two_docstring(comparison="outside")

    def col_vals_in_set(
        df: FrameT,
        column: str,
        set: list[float | int],
        threshold: int = 1,
    ) -> bool:

        compatible_types = COMPATIBLE_TYPES.get("in_set", [])

        return ColValsCompareSet(
            df=df,
            column=column,
            values=set,
            threshold=threshold,
            inside=True,
            allowed_types=compatible_types,
        ).test()

    col_vals_in_set.__doc__ = _col_vals_compare_set_docstring(inside=True)

    def col_vals_not_in_set(
        df: FrameT,
        column: str,
        set: list[float | int],
        threshold: int = 1,
    ) -> bool:

        compatible_types = COMPATIBLE_TYPES.get("not_in_set", [])

        return ColValsCompareSet(
            df=df,
            column=column,
            values=set,
            threshold=threshold,
            inside=False,
            allowed_types=compatible_types,
        ).test()

    col_vals_not_in_set.__doc__ = _col_vals_compare_set_docstring(inside=False)
