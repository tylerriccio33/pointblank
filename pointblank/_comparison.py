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
    x
        The values to compare.
    column
        The column to check when passing a Narwhals DataFrame.
    compare
        The value to compare against. Used in the following comparisons:
        - 'gt' for greater than
        - 'lt' for less than
        - 'eq' for equal to
        - 'ne' for not equal to
        - 'ge' for greater than or equal to
        - 'le' for less than or equal to
    set
        The set of values to compare against. Used in the following comparisons:
        - 'isin' for values in the set
        - 'notin' for values not in the set
    low
        The lower bound of the range of values to compare against. Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    high
        The upper bound of the range of values to compare against. Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    inclusive
        A tuple of booleans that state which bounds are inclusive. The position of the boolean
        corresponds to the value in the following order: (low, high). Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    na_pass
        `True` to pass test units with missing values, `False` otherwise.

    Returns
    -------
    list[bool]
        A list of booleans where `True` indicates a passing test unit.
    """

    x: nw.DataFrame
    column: str = None
    compare: float | int | list[float | int] = None
    set: list[float | int] = None
    low: float | int | list[float | int] = None
    high: float | int | list[float | int] = None
    inclusive: tuple[bool, bool] = None
    na_pass: bool = False

    def gt(self) -> FrameT:

        tbl = (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) > self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

        return tbl

    def lt(self) -> list[bool]:

        tbl = (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) < self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

        return tbl

    def eq(self) -> list[bool]:

        tbl = (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) == self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

        return tbl

    def ne(self) -> list[bool]:

        tbl = (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.when(~nw.col(self.column).is_null())
                .then(nw.col(self.column) != self.compare)
                .otherwise(False),
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

        return tbl

    def ge(self) -> list[bool]:

        tbl = (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) >= self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

        return tbl

    def le(self) -> list[bool]:

        tbl = (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) <= self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

        return tbl

    def between(self) -> list[bool]:

        tbl = self.x

        if self.inclusive == (True, True):
            closed = "both"
        elif self.inclusive == (True, False):
            closed = "left"
        elif self.inclusive == (False, True):
            closed = "right"
        else:
            closed = "none"

        tbl = (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column).is_between(self.low, self.high, closed=closed),
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

        return tbl

    def outside(self) -> list[bool]:

        tbl = self.x

        if self.inclusive == (True, True):
            closed = "both"
        elif self.inclusive == (True, False):
            closed = "left"
        elif self.inclusive == (False, True):
            closed = "right"
        else:
            closed = "none"

        tbl = (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.when(~nw.col(self.column).is_null())
                .then(~nw.col(self.column).is_between(self.low, self.high, closed=closed))
                .otherwise(False),
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

        return tbl

    def isin(self) -> list[bool]:

        tbl = self.x.with_columns(
            pb_is_good_=nw.col(self.column).is_in(self.set),
        ).to_native()

        return tbl

    def notin(self) -> list[bool]:

        tbl = (
            self.x.with_columns(
                pb_is_good_=nw.col(self.column).is_in(self.set),
            )
            .with_columns(pb_is_good_=~nw.col("pb_is_good_"))
            .to_native()
        )

        return tbl


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
                x=dfn,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
            ).gt()
        elif self.comparison == "lt":
            self.test_unit_res = Comparator(
                x=dfn,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
            ).lt()
        elif self.comparison == "eq":
            self.test_unit_res = Comparator(
                x=dfn,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
            ).eq()
        elif self.comparison == "ne":
            self.test_unit_res = Comparator(
                x=dfn,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
            ).ne()
        elif self.comparison == "ge":
            self.test_unit_res = Comparator(
                x=dfn,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
            ).ge()
        elif self.comparison == "le":
            self.test_unit_res = Comparator(
                x=dfn,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
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
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
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
    inclusive : tuple[bool, bool]
        A tuple of booleans that state which bounds are inclusive. The position of the boolean
        corresponds to the value in the following order: (value1, value2).
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
    inclusive: tuple[bool, bool]
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
                x=dfn,
                column=self.column,
                low=self.value1,
                high=self.value2,
                inclusive=self.inclusive,
                na_pass=self.na_pass,
            ).between()
        elif self.comparison == "outside":
            self.test_unit_res = Comparator(
                x=dfn,
                column=self.column,
                low=self.value1,
                high=self.value2,
                inclusive=self.inclusive,
                na_pass=self.na_pass,
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
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
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
            self.test_unit_res = Comparator(x=dfn, column=self.column, set=self.values).isin()
        else:
            self.test_unit_res = Comparator(x=dfn, column=self.column, set=self.values).notin()

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
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
