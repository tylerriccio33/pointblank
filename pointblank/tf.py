from __future__ import annotations

from dataclasses import dataclass

from narwhals.typing import FrameT

from pointblank._constants import COMPATIBLE_DTYPES
from pointblank._constants_docs import ARG_DOCSTRINGS
from pointblank._interrogation import (
    ColValsCompareOne,
    ColValsCompareTwo,
    ColValsCompareSet,
)
from pointblank._utils import _get_assertion_from_fname


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
{ARG_DOCSTRINGS["data"]}
{ARG_DOCSTRINGS["column"]}
{ARG_DOCSTRINGS["value"]}
{ARG_DOCSTRINGS["na_pass"]}
{ARG_DOCSTRINGS["threshold"]}"""


def _col_vals_compare_two_args_docstring() -> str:
    return f"""
Parameters
----------
{ARG_DOCSTRINGS["data"]}
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
{ARG_DOCSTRINGS["data"]}
{ARG_DOCSTRINGS["column"]}
{ARG_DOCSTRINGS["set"]}
{ARG_DOCSTRINGS["threshold"]}"""


@dataclass
class TF:
    """
    Tests with TF functions use tabular data as input and return a single boolean value per check.
    """

    def col_vals_gt(
        data: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        assertion_method = _get_assertion_from_fname()
        compatible_types = COMPATIBLE_DTYPES.get(assertion_method, [])

        return ColValsCompareOne(
            data_tbl=data,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            assertion_method=assertion_method,
            allowed_types=compatible_types,
        ).test()

    col_vals_gt.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="greater than")}
    {_col_vals_compare_one_args_docstring()}
    """

    def col_vals_lt(
        data: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        assertion_method = _get_assertion_from_fname()
        compatible_types = COMPATIBLE_DTYPES.get(assertion_method, [])

        return ColValsCompareOne(
            data_tbl=data,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            assertion_method=assertion_method,
            allowed_types=compatible_types,
        ).test()

    col_vals_lt.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="less than")}
    {_col_vals_compare_one_args_docstring()}
    """

    def col_vals_eq(
        data: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        assertion_method = _get_assertion_from_fname()
        compatible_types = COMPATIBLE_DTYPES.get(assertion_method, [])

        return ColValsCompareOne(
            data_tbl=data,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            assertion_method=assertion_method,
            allowed_types=compatible_types,
        ).test()

    col_vals_eq.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="equal to")}
    {_col_vals_compare_one_args_docstring()}
    """

    def col_vals_ne(
        data: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        assertion_method = _get_assertion_from_fname()
        compatible_types = COMPATIBLE_DTYPES.get(assertion_method, [])

        return ColValsCompareOne(
            data_tbl=data,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            assertion_method=assertion_method,
            allowed_types=compatible_types,
        ).test()

    col_vals_ne.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="not equal to")}
    {_col_vals_compare_one_args_docstring()}
    """

    def col_vals_ge(
        data: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        assertion_method = _get_assertion_from_fname()
        compatible_types = COMPATIBLE_DTYPES.get(assertion_method, [])

        return ColValsCompareOne(
            data_tbl=data,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            assertion_method=assertion_method,
            allowed_types=compatible_types,
        ).test()

    col_vals_ge.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="greater than or equal to")}
    {_col_vals_compare_one_args_docstring()}
    """

    def col_vals_le(
        data: FrameT, column: str, value: float | int, na_pass: bool = False, threshold: int = 1
    ) -> bool:

        assertion_method = _get_assertion_from_fname()
        compatible_types = COMPATIBLE_DTYPES.get(assertion_method, [])

        return ColValsCompareOne(
            data_tbl=data,
            column=column,
            value=value,
            na_pass=na_pass,
            threshold=threshold,
            assertion_method=assertion_method,
            allowed_types=compatible_types,
        ).test()

    col_vals_le.__doc__ = f"""{_col_vals_compare_one_title_docstring(comparison="less than or equal to")}
    {_col_vals_compare_one_args_docstring()}
    """

    def col_vals_between(
        data: FrameT,
        column: str,
        left: float | int,
        right: float | int,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        threshold: int = 1,
    ) -> bool:

        assertion_method = _get_assertion_from_fname()
        compatible_types = COMPATIBLE_DTYPES.get(assertion_method, [])

        return ColValsCompareTwo(
            data_tbl=data,
            column=column,
            value1=left,
            value2=right,
            inclusive=inclusive,
            na_pass=na_pass,
            threshold=threshold,
            assertion_method=assertion_method,
            allowed_types=compatible_types,
        ).test()

    col_vals_between.__doc__ = f"""{_col_vals_compare_two_title_docstring(comparison="between")}
    {_col_vals_compare_two_args_docstring()}
    """

    def col_vals_outside(
        data: FrameT,
        column: str,
        left: float | int,
        right: float | int,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        threshold: int = 1,
    ) -> bool:

        assertion_method = _get_assertion_from_fname()
        compatible_types = COMPATIBLE_DTYPES.get(assertion_method, [])

        return ColValsCompareTwo(
            data_tbl=data,
            column=column,
            value1=left,
            value2=right,
            inclusive=inclusive,
            na_pass=na_pass,
            threshold=threshold,
            assertion_method=assertion_method,
            allowed_types=compatible_types,
        ).test()

    col_vals_outside.__doc__ = f"""{_col_vals_compare_two_title_docstring(comparison="outside of")}
    {_col_vals_compare_two_args_docstring()}
    """

    def col_vals_in_set(
        data: FrameT,
        column: str,
        set: list[float | int],
        threshold: int = 1,
    ) -> bool:

        compatible_types = COMPATIBLE_DTYPES.get("in_set", [])

        return ColValsCompareSet(
            data_tbl=data,
            column=column,
            values=set,
            threshold=threshold,
            inside=True,
            allowed_types=compatible_types,
        ).test()

    col_vals_in_set.__doc__ = f"""{_col_vals_compare_set_title_docstring(inside=True)}
    {_col_vals_compare_set_args_docstring()}
    """

    def col_vals_not_in_set(
        data: FrameT,
        column: str,
        set: list[float | int],
        threshold: int = 1,
    ) -> bool:

        compatible_types = COMPATIBLE_DTYPES.get("not_in_set", [])

        return ColValsCompareSet(
            data_tbl=data,
            column=column,
            values=set,
            threshold=threshold,
            inside=False,
            allowed_types=compatible_types,
        ).test()

    col_vals_not_in_set.__doc__ = f"""{_col_vals_compare_set_title_docstring(inside=False)}
    {_col_vals_compare_set_args_docstring()}
    """
