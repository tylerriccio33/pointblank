from __future__ import annotations
from dataclasses import dataclass

from typing import Any

import narwhals as nw
from narwhals.typing import FrameT

from pointblank._utils import _column_test_prep, _convert_to_narwhals
from pointblank.thresholds import _threshold_check
from pointblank._constants import IBIS_BACKENDS


@dataclass
class Interrogator:
    """
    Compare values against a single value, a set of values, or a range of values.

    Parameters
    ----------
    x
        The values to compare.
    column
        The column to check when passing a Narwhals DataFrame.
    compare
        The value to compare against. Used in the following interrogations:
        - 'gt' for greater than
        - 'lt' for less than
        - 'eq' for equal to
        - 'ne' for not equal to
        - 'ge' for greater than or equal to
        - 'le' for less than or equal to
    set
        The set of values to compare against. Used in the following interrogations:
        - 'isin' for values in the set
        - 'notin' for values not in the set
    pattern
        The regular expression pattern to compare against. Used in the following:
        - 'regex' for values that match the pattern
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
    tbl_type
        The type of table to use for the assertion. This is used to determine the backend for the
        assertion. The default is 'local' but it can also be any of the table types in the
        `IBIS_BACKENDS` constant.

    Returns
    -------
    list[bool]
        A list of booleans where `True` indicates a passing test unit.
    """

    x: nw.DataFrame | Any
    column: str = None
    compare: float | int | list[float | int] = None
    set: list[float | int] = None
    pattern: str = None
    low: float | int | list[float | int] = None
    high: float | int | list[float | int] = None
    inclusive: tuple[bool, bool] = None
    na_pass: bool = False
    tbl_type: str = "local"

    def gt(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column] > ibis.literal(self.compare),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) > self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def lt(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column] < ibis.literal(self.compare),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) < self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def eq(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column] == ibis.literal(self.compare),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) == self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def ne(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=ibis.ifelse(
                    self.x[self.column].notnull(),
                    self.x[self.column] != ibis.literal(self.compare),
                    ibis.literal(False),
                ),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        return (
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

    def ge(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column] >= ibis.literal(self.compare),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) >= self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def le(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column] <= ibis.literal(self.compare),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column) <= self.compare,
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def between(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            low_val = ibis.literal(self.low)
            high_val = ibis.literal(self.high)

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass)
            )

            if self.inclusive[0]:
                tbl = tbl.mutate(pb_is_good_2=tbl[self.column] >= low_val)
            else:
                tbl = tbl.mutate(pb_is_good_2=tbl[self.column] > low_val)

            if self.inclusive[1]:
                tbl = tbl.mutate(pb_is_good_3=tbl[self.column] <= high_val)
            else:
                tbl = tbl.mutate(pb_is_good_3=tbl[self.column] < high_val)

            return tbl.mutate(
                pb_is_good_=tbl.pb_is_good_1 | (tbl.pb_is_good_2 & tbl.pb_is_good_3)
            ).drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")

        closed = _get_nw_closed_str(closed=self.inclusive)

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.col(self.column).is_between(self.low, self.high, closed=closed),
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def outside(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            low_val = ibis.literal(self.low)
            high_val = ibis.literal(self.high)

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass)
            )

            if self.inclusive[0]:
                tbl = tbl.mutate(pb_is_good_2=tbl[self.column] < low_val)
            else:
                tbl = tbl.mutate(pb_is_good_2=tbl[self.column] <= low_val)

            if self.inclusive[1]:
                tbl = tbl.mutate(pb_is_good_3=tbl[self.column] > high_val)
            else:
                tbl = tbl.mutate(pb_is_good_3=tbl[self.column] >= high_val)

            return tbl.mutate(
                pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2 | tbl.pb_is_good_3
            ).drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")

        closed = _get_nw_closed_str(closed=self.inclusive)

        return (
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

    def isin(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            return self.x.mutate(pb_is_good_=self.x[self.column].isin(self.set))

        return self.x.with_columns(
            pb_is_good_=nw.col(self.column).is_in(self.set),
        ).to_native()

    def notin(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            return self.x.mutate(pb_is_good_=self.x[self.column].notin(self.set))

        return (
            self.x.with_columns(
                pb_is_good_=nw.col(self.column).is_in(self.set),
            )
            .with_columns(pb_is_good_=~nw.col("pb_is_good_"))
            .to_native()
        )

    def regex(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            import ibis

            tbl = self.x.mutate(
                pb_is_good_1=self.x[self.column].isnull() & ibis.literal(self.na_pass),
                pb_is_good_2=self.x[self.column].re_search(self.pattern),
            )

            return tbl.mutate(pb_is_good_=tbl.pb_is_good_1 | tbl.pb_is_good_2).drop(
                "pb_is_good_1", "pb_is_good_2"
            )

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.when(~nw.col(self.column).is_null())
                .then(nw.col(self.column).str.contains(pattern=self.pattern))
                .otherwise(False),
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def null(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            return self.x.mutate(
                pb_is_good_=self.x[self.column].isnull(),
            )

        return self.x.with_columns(
            pb_is_good_=nw.col(self.column).is_null(),
        ).to_native()

    def not_null(self) -> FrameT | Any:

        if self.tbl_type in IBIS_BACKENDS:

            return self.x.mutate(
                pb_is_good_=~self.x[self.column].isnull(),
            )

        return self.x.with_columns(
            pb_is_good_=~nw.col(self.column).is_null(),
        ).to_native()


@dataclass
class ColValsCompareOne:
    """
    Compare values in a table column against a single value.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    value
        A value to check against.
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('gt' for greater than, 'lt' for less than).
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    value: float | int
    na_pass: bool
    threshold: int
    assertion_method: str
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.assertion_method == "gt":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).gt()
        elif self.assertion_method == "lt":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).lt()
        elif self.assertion_method == "eq":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).eq()
        elif self.assertion_method == "ne":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).ne()
        elif self.assertion_method == "ge":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).ge()
        elif self.assertion_method == "le":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).le()
        elif self.assertion_method == "null":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                tbl_type=self.tbl_type,
            ).null()
        elif self.assertion_method == "not_null":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                tbl_type=self.tbl_type,
            ).not_null()
        else:
            raise ValueError(
                """Invalid comparison type. Use:
                - `gt` for greater than,
                - `lt` for less than,
                - `eq` for equal to,
                - `ne` for not equal to,
                - `ge` for greater than or equal to,
                - `le` for less than or equal to,
                - `null` for null values, or
                - `not_null` for not null values.
                """
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
    data_tbl
        A data table.
    column
        The column to check.
    value1
        A value to check against.
    value2
        A value to check against.
    inclusive
        A tuple of booleans that state which bounds are inclusive. The position of the boolean
        corresponds to the value in the following order: (value1, value2).
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('between' for between two values and 'outside' for outside two
        values).
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    value1: float | int
    value2: float | int
    inclusive: tuple[bool, bool]
    na_pass: bool
    threshold: int
    assertion_method: str
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.assertion_method == "between":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                low=self.value1,
                high=self.value2,
                inclusive=self.inclusive,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).between()
        elif self.assertion_method == "outside":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                low=self.value1,
                high=self.value2,
                inclusive=self.inclusive,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).outside()
        else:
            raise ValueError(
                """Invalid assertion type. Use:
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
    data_tbl
        A data table.
    column
        The column to check.
    values
        A set of values to check against.
    threshold
        The maximum number of failing test units to allow.
    inside
        `True` to check if the values are inside the set, `False` to check if the values are
        outside the set.
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    values: list[float | int]
    threshold: int
    inside: bool
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.inside:
            self.test_unit_res = Interrogator(
                x=tbl, column=self.column, set=self.values, tbl_type=self.tbl_type
            ).isin()
        else:
            self.test_unit_res = Interrogator(
                x=tbl, column=self.column, set=self.values, tbl_type=self.tbl_type
            ).notin()

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
class ColValsRegex:
    """
    Check if values in a column match a regular expression pattern.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    pattern
        The regular expression pattern to check against.
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    pattern: str
    na_pass: bool
    threshold: int
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        self.test_unit_res = Interrogator(
            x=tbl,
            column=self.column,
            pattern=self.pattern,
            na_pass=self.na_pass,
            tbl_type=self.tbl_type,
        ).regex()

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
class ColExistsHasType:
    """
    Check if a column exists in a DataFrame or has a certain data type.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('exists' for column existence).
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    threshold: int
    assertion_method: str
    tbl_type: str = "local"

    def __post_init__(self):

        if self.tbl_type == "local":

            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _convert_to_narwhals(df=self.data_tbl)

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        if self.tbl_type in IBIS_BACKENDS:
            tbl = self.data_tbl

        if self.assertion_method == "exists":

            res = int(self.column in tbl.columns)

        self.test_unit_res = res

    def get_test_results(self):
        return self.test_unit_res


@dataclass
class NumberOfTestUnits:
    """
    Count the number of test units in a column.
    """

    df: FrameT
    column: str

    def get_test_units(self, tbl_type: str) -> int:

        if tbl_type == "pandas" or tbl_type == "polars":

            # Convert the DataFrame to a format that narwhals can work with and:
            #  - check if the column exists
            dfn = _column_test_prep(
                df=self.df, column=self.column, allowed_types=None, check_exists=False
            )

            return len(dfn)

        if tbl_type in IBIS_BACKENDS:

            # Get the count of test units and convert to a native format
            # TODO: check whether pandas or polars is available
            return self.df.count().to_polars()


def _get_nw_closed_str(closed: tuple[bool, bool]) -> str:
    """
    Get the string representation of the closed bounds for the `is_between` method in Narwhals.
    """

    if closed == (True, True):
        return "both"
    elif closed == (True, False):
        return "left"
    elif closed == (False, True):
        return "right"
    else:
        return "none"
