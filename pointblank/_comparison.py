from __future__ import annotations
from dataclasses import dataclass

import narwhals as nw
from narwhals.typing import FrameT

from pointblank._utils import _column_test_prep
from pointblank.thresholds import _threshold_check


@dataclass
class Comparator:
    x: float | int | list[float | int] | nw.DataFrame
    column: str = None
    compare: float | int | list[float | int] = None
    low: float | int | list[float | int] = None
    high: float | int | list[float | int] = None

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
        return [i > j for i, j in zip(self.x, self.compare)]

    def lt(self) -> list[bool]:
        return [i < j for i, j in zip(self.x, self.compare)]

    def eq(self) -> list[bool]:
        return [i == j for i, j in zip(self.x, self.compare)]

    def ne(self) -> list[bool]:
        return [i != j for i, j in zip(self.x, self.compare)]

    def ge(self) -> list[bool]:
        return [i >= j for i, j in zip(self.x, self.compare)]

    def le(self) -> list[bool]:
        return [i <= j for i, j in zip(self.x, self.compare)]

    def between(self) -> list[bool]:
        return [i > j and i < k for i, j, k in zip(self.x, self.low, self.high)]

    def outside(self) -> list[bool]:
        return [i < j or i > k for i, j, k in zip(self.x, self.low, self.high)]

    def isin(self) -> list[bool]:
        return [i in self.compare for i in self.x]

    def notin(self) -> list[bool]:
        return [i not in self.compare for i in self.x]

    def isnull(self) -> list[bool]:
        return [i is None for i in self.x]

    def notnull(self) -> list[bool]:
        return [i is not None for i in self.x]


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
            self.test_unit_res = Comparator(x=dfn, column=self.column, compare=self.value).gt()
        elif self.comparison == "lt":
            self.test_unit_res = Comparator(x=dfn, column=self.column, compare=self.value).lt()
        elif self.comparison == "eq":
            self.test_unit_res = Comparator(x=dfn, column=self.column, compare=self.value).eq()
        elif self.comparison == "ne":
            self.test_unit_res = Comparator(x=dfn, column=self.column, compare=self.value).ne()
        elif self.comparison == "ge":
            self.test_unit_res = Comparator(x=dfn, column=self.column, compare=self.value).ge()
        elif self.comparison == "le":
            self.test_unit_res = Comparator(x=dfn, column=self.column, compare=self.value).le()
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
                x=dfn, column=self.column, low=self.value1, high=self.value2
            ).between()
        elif self.comparison == "outside":
            self.test_unit_res = Comparator(
                x=dfn, column=self.column, low=self.value1, high=self.value2
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
