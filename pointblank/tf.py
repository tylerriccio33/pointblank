from __future__ import annotations

from dataclasses import dataclass

from narwhals.typing import FrameT

from pointblank._constants import COMPATIBLE_TYPES
from pointblank._constants_docs import ARG_DOCSTRINGS
from pointblank._comparison import (
    ColValsCompareOne,
    ColValsCompareTwo,
    ColValsCompareSet,
)
from pointblank._utils import _get_comparison_from_fname


def _col_vals_compare_one_title_docstring(comparison: str) -> str:
    return "Test whether column values are ___ a single value.".replace("___", comparison)


def _col_vals_compare_two_title_docstring(comparison: str) -> str:
    return "Test whether column values are ___ two values.".replace("___", comparison)


def _col_vals_compare_set_title_docstring(inside: bool) -> str:
    return "Test whether column values are ___ a set of values.".replace(
        "___", "in" if inside else "not in"
    )


def _col_vals_compare_one_args_docstring() -> str:
    return f"""
Parameters
----------
{ARG_DOCSTRINGS["df"]}
{ARG_DOCSTRINGS["column"]}
{ARG_DOCSTRINGS["value"]}
{ARG_DOCSTRINGS["na_pass"]}
{ARG_DOCSTRINGS["threshold"]}"""


def _col_vals_compare_two_args_docstring() -> str:
    return f"""
Parameters
----------
{ARG_DOCSTRINGS["df"]}
{ARG_DOCSTRINGS["column"]}
{ARG_DOCSTRINGS["left"]}
{ARG_DOCSTRINGS["right"]}
{ARG_DOCSTRINGS["inclusive"]}
{ARG_DOCSTRINGS["na_pass"]}
{ARG_DOCSTRINGS["threshold"]}"""


def _col_vals_compare_set_args_docstring() -> str:
    return f"""
Parameters
----------
{ARG_DOCSTRINGS["df"]}
{ARG_DOCSTRINGS["column"]}
{ARG_DOCSTRINGS["set"]}
{ARG_DOCSTRINGS["threshold"]}"""


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
            compare_strategy="list",
        ).test()

    col_vals_gt.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="greater than")}
    {_col_vals_compare_one_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_lt.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="less than")}
    {_col_vals_compare_one_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_eq.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="equal to")}
    {_col_vals_compare_one_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_ne.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="not equal to")}
    {_col_vals_compare_one_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_ge.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="greater than or equal to")}
    {_col_vals_compare_one_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_le.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="less than or equal to")}
    {_col_vals_compare_one_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_between.__doc__ = f"""{_col_vals_compare_two_title_docstring(comparison="between")}
    {_col_vals_compare_two_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_outside.__doc__ = f"""{_col_vals_compare_two_title_docstring(comparison="outside of")}
    {_col_vals_compare_two_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_in_set.__doc__ = f"""{_col_vals_compare_set_title_docstring(inside=True)}
    {_col_vals_compare_set_args_docstring()}
    """

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
            compare_strategy="list",
        ).test()

    col_vals_not_in_set.__doc__ = f"""{_col_vals_compare_set_title_docstring(inside=False)}
    {_col_vals_compare_set_args_docstring()}
    """
