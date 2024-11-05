from __future__ import annotations
from dataclasses import dataclass

import operator

import narwhals as nw
from narwhals.typing import FrameT

from pointblank._utils import _column_test_prep, _get_def_name
from pointblank.thresholds import _threshold_check


@dataclass
class Comparator:
    """
    Compare values against a single value, a set of values, or a range of values.

    Parameters
    ----------
    x : float | int | list[float | int] | nw.DataFrame
        The values to compare.
    column : str
        The column to check when passing a Narwhals DataFrame.
    compare : float | int | list[float | int]
        The value to compare against. Used in the following comparisons:
        - 'gt' for greater than
        - 'lt' for less than
        - 'eq' for equal to
        - 'ne' for not equal to
        - 'ge' for greater than or equal to
        - 'le' for less than or equal to
    set : list[float | int]
        The set of values to compare against. Used in the following comparisons:
        - 'isin' for values in the set
        - 'notin' for values not in the set
    low : float | int | list[float | int]
        The lower bound of the range of values to compare against. Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    high : float | int | list[float | int]
        The upper bound of the range of values to compare against. Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    na_pass : bool
        `True` to pass test units with missing values, `False` otherwise.

    Returns
    -------
    list[bool]
        A list of booleans where `True` indicates a passing test unit.
    """

    x: float | int | list[float | int] | nw.DataFrame
    column: str = None
    compare: float | int | list[float | int] = None
    set: list[float | int] = None
    low: float | int | list[float | int] = None
    high: float | int | list[float | int] = None
    na_pass: bool = False

    def __post_init__(self):

        if isinstance(self.x, nw.DataFrame):

            if self.column is None:
                raise ValueError("A column must be provided when passing a Narwhals DataFrame.")

            self.x = self.x[self.column].to_list()

        elif not isinstance(self.x, list):
            self.x = [self.x]

        if self.compare is not None:
            self.compare = self._ensure_list(self.compare, len(self.x), "compare")

        if self.low is not None:
            self.low = self._ensure_list(self.low, len(self.x), "low")

        if self.high is not None:
            self.high = self._ensure_list(self.high, len(self.x), "high")

    def _ensure_list(self, value, length=None, name=None):
        if not isinstance(value, list):
            value = [value] * (length if length is not None else 1)
        elif length is not None and len(value) != length:
            raise ValueError(f"Length of `x` and `{name}` must be the same.")

        return value

    def gt(self) -> list[bool]:

        op = _get_def_name()
        return [
            _compare_values(i, j, op=op, na_pass=self.na_pass) for i, j in zip(self.x, self.compare)
        ]

    def lt(self) -> list[bool]:

        op = _get_def_name()
        return [
            _compare_values(i, j, op=op, na_pass=self.na_pass) for i, j in zip(self.x, self.compare)
        ]

    def eq(self) -> list[bool]:

        op = _get_def_name()
        return [
            _compare_values(i, j, op=op, na_pass=self.na_pass) for i, j in zip(self.x, self.compare)
        ]

    def ne(self) -> list[bool]:

        op = _get_def_name()
        return [
            _compare_values(i, j, op=op, na_pass=self.na_pass) for i, j in zip(self.x, self.compare)
        ]

    def ge(self) -> list[bool]:

        op = _get_def_name()
        return [
            _compare_values(i, j, op=op, na_pass=self.na_pass) for i, j in zip(self.x, self.compare)
        ]

    def le(self) -> list[bool]:

        op = _get_def_name()
        return [
            _compare_values(i, j, op=op, na_pass=self.na_pass) for i, j in zip(self.x, self.compare)
        ]

    def between(self) -> list[bool]:
        return [
            _compare_values_range(i, j, k, between=True, na_pass=self.na_pass)
            for i, j, k in zip(self.x, self.low, self.high)
        ]

    def outside(self) -> list[bool]:
        return [
            _compare_values_range(i, j, k, between=False, na_pass=self.na_pass)
            for i, j, k in zip(self.x, self.low, self.high)
        ]

    def isin(self) -> list[bool]:
        return [i in self.set for i in self.x]

    def notin(self) -> list[bool]:
        return [i not in self.set for i in self.x]

    def isnull(self) -> list[bool]:
        return [i is None for i in self.x]

    def notnull(self) -> list[bool]:
        return [i is not None for i in self.x]


def _compare_values(i, j, op, na_pass):
    """
    Compare two values using the specified operator.

    Parameters
    ----------
    i : int | float
        The first value.
    j : int | float
        The second value.
    op : str
        The operator as a string. Supported operators are:
        - 'gt' for greater than
        - 'lt' for less than
        - 'eq' for equal to
        - 'ne' for not equal to
        - 'ge' for greater than or equal to
        - 'le' for less than or equal to

    Returns
    -------
    bool
        The result of the comparison.
    """
    ops = {
        "gt": operator.gt,
        "lt": operator.lt,
        "eq": operator.eq,
        "ne": operator.ne,
        "ge": operator.ge,
        "le": operator.le,
    }

    if op not in ops:
        raise ValueError(f"Unsupported operator: {op}")

    if i is None:
        return na_pass

    return ops[op](i, j)


def _compare_values_range(
    i: int | float | None, j: int | float, k: int | float, between: bool, na_pass: bool
):
    """
    Compare a value against a range of values using the specified operator.

    Parameters
    ----------
    i : int | float
        The value to compare.
    j : int | float
        The lower bound of the range.
    k : int | float
        The upper bound of the range.
    between : bool
        `True` to check if the value is between the range, `False` to check if the value is outside
        the range.
    na_pass : bool
        `True` to pass test units with missing values, `False` otherwise.

    Returns
    -------
    bool
        The result of the comparison.
    """

    if i is None:
        return na_pass

    if between:
        return operator.gt(i, j) and operator.lt(i, k)
    else:
        return operator.lt(i, j) or operator.gt(i, k)


@dataclass
class ColValsCompareOne:
    """
    Compare values in a table column against a single value.

    Parameters
    ----------
    df : FrameT
        a DataFrame.
    column : str
        The column to check.
    value : float | int
        A value to check against.
    na_pass: bool
        `True` to pass test units with missing values, `False` otherwise.
    threshold : int
        The maximum number of failing test units to allow.
    comparison : str
        The type of comparison ('gt' for greater than, 'lt' for less than).
    allowed_types : list[str]
        The allowed data types for the column.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    df: FrameT
    column: str
    value: float | int
    na_pass: bool
    threshold: int
    comparison: str
    allowed_types: list[str]

    def __post_init__(self):

        # Convert the DataFrame to a format that narwhals can work with and:
        #  - check if the column exists
        #  - check if the column type is compatible with the test
        dfn = _column_test_prep(df=self.df, column=self.column, allowed_types=self.allowed_types)

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.comparison == "gt":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, compare=self.value, na_pass=self.na_pass
            ).gt()
        elif self.comparison == "lt":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, compare=self.value, na_pass=self.na_pass
            ).lt()
        elif self.comparison == "eq":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, compare=self.value, na_pass=self.na_pass
            ).eq()
        elif self.comparison == "ne":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, compare=self.value, na_pass=self.na_pass
            ).ne()
        elif self.comparison == "ge":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, compare=self.value, na_pass=self.na_pass
            ).ge()
        elif self.comparison == "le":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, compare=self.value, na_pass=self.na_pass
            ).le()
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

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` and then determine
        # if the test passes overall by comparing the number of failing test units to the threshold
        # for failing test units
        return _threshold_check(
            failing_test_units=self.test_unit_res.count(False), threshold=self.threshold
        )


@dataclass
class ColValsCompareTwo:
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
    na_pass: bool
        `True` to pass test units with missing values, `False` otherwise.
    threshold : int
        The maximum number of failing test units to allow.
    comparison : str
        The type of comparison ('between' for between two values and 'outside' for outside two
        values).
    allowed_types : list[str]
        The allowed data types for the column.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    df: FrameT
    column: str
    value1: float | int
    value2: float | int
    na_pass: bool
    threshold: int
    comparison: str
    allowed_types: list[str]

    def __post_init__(self):

        # Convert the DataFrame to a format that narwhals can work with and:
        #  - check if the column exists
        #  - check if the column type is compatible with the test
        dfn = _column_test_prep(df=self.df, column=self.column, allowed_types=self.allowed_types)

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.comparison == "between":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, low=self.value1, high=self.value2, na_pass=self.na_pass
            ).between()
        elif self.comparison == "outside":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, low=self.value1, high=self.value2, na_pass=self.na_pass
            ).outside()
        else:
            raise ValueError(
                """Invalid comparison type. Use:
                - `between` for values between two values, or
                - `outside` for values outside two values."""
            )

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` and then determine
        # if the test passes overall by comparing the number of failing test units to the threshold
        # for failing test units
        return _threshold_check(
            failing_test_units=self.test_unit_res.count(False), threshold=self.threshold
        )


@dataclass
class ColValsCompareSet:
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
    inside : bool
        `True` to check if the values are inside the set, `False` to check if the values are
        outside the set.
    allowed_types : list[str]
        The allowed data types for the column.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    df: FrameT
    column: str
    values: list[float | int]
    threshold: int
    inside: bool
    allowed_types: list[str]

    def __post_init__(self):

        # Convert the DataFrame to a format that narwhals can work with and:
        #  - check if the column exists
        #  - check if the column type is compatible with the test
        dfn = _column_test_prep(df=self.df, column=self.column, allowed_types=self.allowed_types)

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.inside:
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, compare=self.values
            ).between()
        else:
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, compare=self.values
            ).outside()

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` and then determine
        # if the test passes overall by comparing the number of failing test units to the threshold
        # for failing test units
        return _threshold_check(
            failing_test_units=self.test_unit_res.count(False), threshold=self.threshold
        )


@dataclass
class NumberOfTestUnits:
    """
    Count the number of test units in a column.
    """

    df: FrameT
    column: str

    def __post_init__(self):

        # Convert the DataFrame to a format that narwhals can work with and:
        #  - check if the column exists
        dfn = _column_test_prep(df=self.df, column=self.column, allowed_types=["numeric"])

        self.test_units = len(dfn)

    def get_test_units(self):
        return self.test_units
