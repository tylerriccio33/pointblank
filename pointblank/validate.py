from __future__ import annotations

import datetime
import json
import re

from dataclasses import dataclass, field

import narwhals as nw
from narwhals.typing import FrameT
from great_tables import GT, md, html, loc, style, google_font

from pointblank._constants import (
    TYPE_METHOD_MAP,
    COMPATIBLE_TYPES,
    COMPARE_TYPE_MAP,
    VALIDATION_REPORT_FIELDS,
    SVG_ICONS_FOR_ASSERTION_TYPES,
)
from pointblank._comparison import (
    ColValsCompareOne,
    ColValsCompareTwo,
    ColValsCompareSet,
    NumberOfTestUnits,
)
from pointblank._utils import _get_def_name, _check_invalid_fields
from pointblank.thresholds import (
    Thresholds,
    _normalize_thresholds_creation,
    _convert_abs_count_to_fraction,
)


COL_VALS_GT_TITLE_DOCSTRING = """
    Validate whether column values are greater than a fixed value.
    """

COL_VALS_LT_TITLE_DOCSTRING = """
    Validate whether column values are less than a fixed value.
    """

COL_VALS_EQ_TITLE_DOCSTRING = """
    Validate whether column values are equal to a fixed value.
    """

COL_VALS_NE_TITLE_DOCSTRING = """
    Validate whether column values are not equal to a fixed value.
    """

COL_VALS_GE_TITLE_DOCSTRING = """
    Validate whether column values are greater than or equal to a fixed value.
    """

COL_VALS_LE_TITLE_DOCSTRING = """
    Validate whether column values are less than or equal to a fixed value.
    """

COL_VALS_BETWEEN_TITLE_DOCSTRING = """
    Validate whether column values are between two values.
    """

COL_VALS_OUTSIDE_TITLE_DOCSTRING = """
    Validate whether column values are outside of a range.
    """

COL_VALS_IN_SET_TITLE_DOCSTRING = """
    Validate whether column values are in a set of values.
    """

COL_VALS_NOT_IN_SET_TITLE_DOCSTRING = """
    Validate whether column values are not in a set of values.
    """

COL_VALS_COMPARE_ONE_PARAMETERS_DOCSTRING = """
    Parameters
    ----------
    column : str
        The column to validate.
    value : int | float
        The value to compare against.
    na_pass : bool
        Whether to pass rows with missing values.
    thresholds : int | float | tuple | dict| Thresholds, optional
        The threshold value or values.
    active : bool, optional
        Whether the validation is active.
    """

COL_VALS_COMPARE_TWO_PARAMETERS_DOCSTRING = """
    Parameters
    ----------
    column : str
        The column to validate.
    left : int | float
        The lower bound of the range.
    right : int | float
        The upper bound of the range.
    na_pass : bool
        Whether to pass rows with missing values.
    thresholds : int | float | tuple | dict| Thresholds, optional
        The threshold value or values.
    active : bool, optional
        Whether the validation is active.
    """

COL_VALS_COMPARE_SET_PARAMETERS_DOCSTRING = """
    Parameters
    ----------
    column : str
        The column to validate.
    set : list[int | float]
        A list of values to compare against.
    thresholds : int | float | tuple | dict| Thresholds, optional
        The threshold value or values.
    active : bool, optional
        Whether the validation is active.
    """


@dataclass
class _ValidationInfo:
    """
    Information about a validation to be performed on a table and the results of the interrogation.

    Attributes
    ----------
    i : int | None
        The validation step number.
    i_o : int | None
        The original validation step number (if a step creates multiple steps). Unused.
    step_id : str | None
        The ID of the step (if a step creates multiple steps). Unused.
    sha1 : str | None
        The SHA-1 hash of the step. Unused.
    assertion_type : str | None
        The type of assertion. This is the method name of the validation (e.g., `"col_vals_gt"`).
    column : str | None
        The column to validate. Currently we don't allow for column expressions (which may map to
        multiple columns).
    values : any | list[any] | tuple | None
        The value or values to compare against.
    na_pass : bool | None
        Whether to pass test units that hold missing values.
    thresholds : Thresholds | None
        The threshold values for the validation.
    label : str | None
        A label for the validation step. Unused.
    brief : str | None
        A brief description of the validation step. Unused.
    active : bool | None
        Whether the validation step is active.
    all_passed : bool | None
        Upon interrogation, this describes whether all test units passed for a validation step.
    n : int | None
        The number of test units for the validation step.
    n_passed : int | None
        The number of test units that passed (i.e., passing test units).
    n_failed : int | None
        The number of test units that failed (i.e., failing test units).
    f_passed : int | None
        The fraction of test units that passed. The calculation is `n_passed / n`.
    f_failed : int | None
        The fraction of test units that failed. The calculation is `n_failed / n`.
    warn : bool | None
        Whether the number of failing test units is beyond the warning threshold.
    stop : bool | None
        Whether the number of failing test units is beyond the stopping threshold.
    notify : bool | None
        Whether the number of failing test units is beyond the notification threshold.
    row_sample : int | None
        The number of rows to sample for the validation step. Unused.
    tbl_checked : bool | None
        The data table in its native format that has been checked for the validation step. It wil
        include a new column called `pb_is_good_` that is a boolean column that indicates whether
        the row passed the validation or not.
    time_processed : str | None
        The time the validation step was processed. This is in the ISO 8601 format in UTC time.
    proc_duration_s : float | None
        The duration of processing for the validation step in seconds.
    """

    # Validation plan
    i: int | None = None
    i_o: int | None = None
    step_id: str | None = None
    sha1: str | None = None
    assertion_type: str | None = None
    column: str | None = None
    values: any | list[any] | tuple | None = None
    inclusive: tuple[bool, bool] | None = None
    na_pass: bool | None = None
    thresholds: Thresholds | None = None
    label: str | None = None
    brief: str | None = None
    active: bool | None = None
    # Interrogation results
    all_passed: bool | None = None
    n: int | None = None
    n_passed: int | None = None
    n_failed: int | None = None
    f_passed: int | None = None
    f_failed: int | None = None
    warn: bool | None = None
    stop: bool | None = None
    notify: bool | None = None
    row_sample: int | None = None
    tbl_checked: bool | None = None
    time_processed: str | None = None
    proc_duration_s: float | None = None


@dataclass
class Validate:
    data: FrameT
    tbl_name: str | None = None
    label: str | None = None
    thresholds: int | float | tuple | dict | Thresholds | None = None

    """
    A container for a table and a set of validations to be performed on the table.

    Parameters
    ----------
    data : FrameT
        The table to validate. Can be a Pandas or a Polars DataFrame.
    tbl_name : str | None, optional
        A optional name to assign to the input table object. If no value is provided, a name will
        be generated based on whatever information is available. This table name will be displayed
        in the header area of the HTML report generated by using the `report_as_html()` method.
    label : str | None, optional
        An optional label for the validation plan. If no value is provided, a label will be
        generated based on the current system date and time. Markdown can be used here to make the
        label more visually appealing (it will appear in the header area of the HTML report).
    thresholds : int | float | tuple | dict| Thresholds, optional
        Generate threshold failure levels so that all validation steps can report and react
        accordingly when exceeding the set levels. This is to be created in one of several input
        schemes: (1) single integer/float denoting absolute number or fraction of failing test units
        for the 'warn' level, (2) a tuple of 1-3 values, (3) a dictionary of 1-3 entries, or a
        Thresholds object.

    Returns
    -------
    Validate
        A `Validate` object with the table and validations to be performed.
    """

    def __post_init__(self):

        # Check input of the `thresholds=` argument
        _check_thresholds(thresholds=self.thresholds)

        # Normalize the thresholds value (if any) to a Thresholds object
        self.thresholds = _normalize_thresholds_creation(self.thresholds)

        # TODO: Add functionality to obtain the column names and types from the table
        self.col_names = None
        self.col_types = None

        # TODO: add the starting and ending datetime values for the interrogation
        self.time_start = None
        self.time_end = None

        self.validation_info = []

    def col_vals_gt(
        self,
        column: str,
        value: float | int,
        na_pass: bool = False,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_value_float_int(value=value)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            na_pass=na_pass,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_gt.__doc__ = COL_VALS_GT_TITLE_DOCSTRING + COL_VALS_COMPARE_ONE_PARAMETERS_DOCSTRING

    def col_vals_lt(
        self,
        column: str,
        value: float | int,
        na_pass: bool = False,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_value_float_int(value=value)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            na_pass=na_pass,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_lt.__doc__ = COL_VALS_LT_TITLE_DOCSTRING + COL_VALS_COMPARE_ONE_PARAMETERS_DOCSTRING

    def col_vals_eq(
        self,
        column: str,
        value: float | int,
        na_pass: bool = False,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_value_float_int(value=value)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            na_pass=na_pass,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_eq.__doc__ = COL_VALS_EQ_TITLE_DOCSTRING + COL_VALS_COMPARE_ONE_PARAMETERS_DOCSTRING

    def col_vals_ne(
        self,
        column: str,
        value: float | int,
        na_pass: bool = False,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_value_float_int(value=value)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            na_pass=na_pass,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_ne.__doc__ = COL_VALS_NE_TITLE_DOCSTRING + COL_VALS_COMPARE_ONE_PARAMETERS_DOCSTRING

    def col_vals_ge(
        self,
        column: str,
        value: float | int,
        na_pass: bool = False,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_value_float_int(value=value)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            na_pass=na_pass,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_ge.__doc__ = COL_VALS_GE_TITLE_DOCSTRING + COL_VALS_COMPARE_ONE_PARAMETERS_DOCSTRING

    def col_vals_le(
        self,
        column: str,
        value: float | int,
        na_pass: bool = False,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_value_float_int(value=value)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            na_pass=na_pass,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_le.__doc__ = COL_VALS_LE_TITLE_DOCSTRING + COL_VALS_COMPARE_ONE_PARAMETERS_DOCSTRING

    def col_vals_between(
        self,
        column: str,
        left: float | int,
        right: float | int,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_value_float_int(value=left)
        _check_value_float_int(value=right)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        value = (left, right)

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            inclusive=inclusive,
            na_pass=na_pass,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_between.__doc__ = (
        COL_VALS_BETWEEN_TITLE_DOCSTRING + COL_VALS_COMPARE_TWO_PARAMETERS_DOCSTRING
    )

    def col_vals_outside(
        self,
        column: str,
        left: float | int,
        right: float | int,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_value_float_int(value=left)
        _check_value_float_int(value=right)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=na_pass, param_name="na_pass")
        _check_boolean_input(param=active, param_name="active")

        value = (left, right)

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            inclusive=inclusive,
            na_pass=na_pass,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_outside.__doc__ = (
        COL_VALS_OUTSIDE_TITLE_DOCSTRING + COL_VALS_COMPARE_TWO_PARAMETERS_DOCSTRING
    )

    def col_vals_in_set(
        self,
        column: str,
        set: list[float | int],
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_set_types(set=set)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=active, param_name="active")

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=set,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_in_set.__doc__ = (
        COL_VALS_IN_SET_TITLE_DOCSTRING + COL_VALS_COMPARE_SET_PARAMETERS_DOCSTRING
    )

    def col_vals_not_in_set(
        self,
        column: str,
        set: list[float | int],
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        assertion_type = _get_def_name()

        _check_column(column=column)
        _check_set_types(set=set)
        _check_thresholds(thresholds=thresholds)
        _check_boolean_input(param=active, param_name="active")

        val_info = _ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=set,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        return self._add_validation(validation_info=val_info)

    col_vals_not_in_set.__doc__ = (
        COL_VALS_NOT_IN_SET_TITLE_DOCSTRING + COL_VALS_COMPARE_SET_PARAMETERS_DOCSTRING
    )

    def interrogate(self):
        """
        Evaluate each validation against the table and store the results.

        Returns
        -------
        Validate
            The `Validate` object with the results of the interrogation.
        """

        df = self.data

        for validation in self.validation_info:

            start_time = datetime.datetime.now(datetime.timezone.utc)

            # Make a copy of the table for this step
            df_step = df

            type = validation.assertion_type
            column = validation.column
            value = validation.values
            inclusive = validation.inclusive
            na_pass = validation.na_pass
            threshold = validation.thresholds

            comparison = TYPE_METHOD_MAP[type]
            compare_type = COMPARE_TYPE_MAP[comparison]
            compatible_types = COMPATIBLE_TYPES.get(comparison, [])

            validation.n = NumberOfTestUnits(df=df_step, column=column).get_test_units()

            if compare_type == "COMPARE_ONE":

                results_tbl = ColValsCompareOne(
                    df=df_step,
                    column=column,
                    value=value,
                    na_pass=na_pass,
                    threshold=threshold,
                    comparison=comparison,
                    allowed_types=compatible_types,
                    compare_strategy="table",
                ).get_test_results()

            if compare_type == "COMPARE_TWO":

                results_tbl = ColValsCompareTwo(
                    df=df_step,
                    column=column,
                    value1=value[0],
                    value2=value[1],
                    inclusive=inclusive,
                    na_pass=na_pass,
                    threshold=threshold,
                    comparison=comparison,
                    allowed_types=compatible_types,
                    compare_strategy="table",
                ).get_test_results()

            if compare_type == "COMPARE_SET":

                inside = True if comparison == "in_set" else False

                results_tbl = ColValsCompareSet(
                    df=df_step,
                    column=column,
                    values=value,
                    threshold=threshold,
                    inside=inside,
                    allowed_types=compatible_types,
                    compare_strategy="table",
                ).get_test_results()

            # Extract the `pb_is_good_` column from the table as a results list
            results_list = nw.from_native(results_tbl)["pb_is_good_"].to_list()

            validation.all_passed = all(results_list)
            validation.n = len(results_list)
            validation.n_passed = results_list.count(True)
            validation.n_failed = results_list.count(False)

            # Calculate fractions of passing and failing test units
            # - `f_passed` is the fraction of test units that passed
            # - `f_failed` is the fraction of test units that failed
            for attr in ["passed", "failed"]:
                setattr(
                    validation,
                    f"f_{attr}",
                    _convert_abs_count_to_fraction(
                        value=getattr(validation, f"n_{attr}"), test_units=validation.n
                    ),
                )

            # Determine if the number of failing test units is beyond the threshold value
            # for each of the severity levels
            # - `warn` is the threshold for a warning
            # - `stop` is the threshold for stopping
            # - `notify` is the threshold for notifying
            for level in ["warn", "stop", "notify"]:
                setattr(
                    validation,
                    level,
                    threshold._threshold_result(
                        fraction_failing=validation.f_failed, test_units=validation.n, level=level
                    ),
                )

            # Include the results table that has a new column called `pb_is_good_`; that
            # is a boolean column that indicates whether the row passed the validation or not
            validation.tbl_checked = results_tbl

            # Get the end time for this step
            end_time = datetime.datetime.now(datetime.timezone.utc)

            # Calculate the duration of processing for this step
            validation.proc_duration_s = (end_time - start_time).total_seconds()

            # Set the time of processing for this step, this should be UTC time is ISO 8601 format
            validation.time_processed = end_time.isoformat(timespec="milliseconds")

        return self

    def all_passed(self):
        """
        Determine if every validation step passed perfectly, with no failing test units.

        Returns
        -------
        bool
            `True` if all validations passed, `False` otherwise.
        """
        return all(validation.all_passed for validation in self.validation_info)

    def n(self, i: int | list[int] | None = None):
        """
        Provides a dictionary of the number of test units for each validation step.

        Parameters
        ----------
        i : int | list[int], optional
            The validation step number(s) from which the number of test units is obtained.
            If `None`, all steps are included.

        Returns
        -------
        dict[int, int]
            A dictionary of the number of test units for each validation step.
        """

        return self._get_validation_dict(i, "n")

    def n_passed(self, i: int | list[int] | None = None):
        """
        Provides a dictionary of the number of test units that passed for each validation step.

        Parameters
        ----------
        i : int | list[int], optional
            The validation step number(s) from which the number of passing test units is obtained.
            If `None`, all steps are included.

        Returns
        -------
        dict[int, int]
            A dictionary of the number of failing test units for each validation step.
        """

        return self._get_validation_dict(i, "n_passed")

    def n_failed(self, i: int | list[int] | None = None):
        """
        Provides a dictionary of the number of test units that failed for each validation step.

        Parameters
        ----------
        i : int | list[int], optional
            The validation step number(s) from which the number of failing test units is obtained.
            If `None`, all steps are included.

        Returns
        -------
        dict[int, int]
            A dictionary of the number of failing test units for each validation step.
        """

        return self._get_validation_dict(i, "n_failed")

    def f_passed(self, i: int | list[int] | None = None):
        """
        Provides a dictionary of the fraction of test units that passed for each validation step.

        Parameters
        ----------
        i : int | list[int], optional
            The validation step number(s) from which the fraction of passing test units is obtained.
            If `None`, all steps are included.

        Returns
        -------
        dict[int, float]
            A dictionary of the fraction of passing test units for each validation step.
        """

        return self._get_validation_dict(i, "f_passed")

    def f_failed(self, i: int | list[int] | None = None):
        """
        Provides a dictionary of the fraction of test units that failed for each validation step.

        Parameters
        ----------
        i : int | list[int], optional
            The validation step number(s) from which the fraction of failing test units is obtained.
            If `None`, all steps are included.

        Returns
        -------
        dict[int, float]
            A dictionary of the fraction of failing test units for each validation step.
        """

        return self._get_validation_dict(i, "f_failed")

    def warn(self, i: int | list[int] | None = None):
        """
        Provides a dictionary of the warning status for each validation step.

        Parameters
        ----------
        i : int | list[int], optional
            The validation step number(s) from which the warning status is obtained.
            If `None`, all steps are included.

        Returns
        -------
        dict[int, bool]
            A dictionary of the warning status for each validation step.
        """

        return self._get_validation_dict(i, "warn")

    def stop(self, i: int | list[int] | None = None):
        """
        Provides a dictionary of the stopping status for each validation step.

        Parameters
        ----------
        i : int | list[int], optional
            The validation step number(s) from which the stopping status is obtained.
            If `None`, all steps are included.

        Returns
        -------
        dict[int, bool]
            A dictionary of the stopping status for each validation step.
        """

        return self._get_validation_dict(i, "stop")

    def notify(self, i: int | list[int] | None = None):
        """
        Provides a dictionary of the notification status for each validation step.

        Parameters
        ----------
        i : int | list[int], optional
            The validation step number(s) from which the notification status is obtained.
            If `None`, all steps are included.

        Returns
        -------
        dict[int, bool]
            A dictionary of the notification status for each validation step.
        """

        return self._get_validation_dict(i, "notify")

    def report_as_json(
        self, use_fields: list[str] | None = None, exclude_fields: list[str] | None = None
    ) -> str:
        """
        Get a report of the validation results.

        Parameters
        ----------
        use_fields : list[str], optional
            A list of fields to include in the report. If `None`, all fields are included.
        exclude_fields : list[str], optional
            A list of fields to exclude from the report. If `None`, no fields are excluded.

        Returns
        -------
        str
            A JSON-formatted string representing the validation report.
        """

        if use_fields is not None and exclude_fields is not None:
            raise ValueError("Cannot specify both `use_fields=` and `exclude_fields=`.")

        if use_fields is None:
            fields = VALIDATION_REPORT_FIELDS
        else:

            # Ensure that the fields to use are valid
            _check_invalid_fields(use_fields, VALIDATION_REPORT_FIELDS)

            fields = use_fields

        if exclude_fields is not None:

            # Ensure that the fields to exclude are valid
            _check_invalid_fields(exclude_fields, VALIDATION_REPORT_FIELDS)

            fields = [field for field in fields if field not in exclude_fields]

        report = []

        for validation_info in self.validation_info:
            report_entry = {
                field: getattr(validation_info, field) for field in VALIDATION_REPORT_FIELDS
            }

            # Filter the report entry based on the fields to include
            report_entry = {field: report_entry[field] for field in fields}

            report.append(report_entry)

        return json.dumps(report, indent=4, default=str)

    def report_as_html(self) -> GT:
        """
        Validation report builder: generates an HTML table using Great Tables
        """

        # Determine whether pandas or polars is available
        try:
            import pandas as pd
        except ImportError:
            pd = None

        try:
            import polars as pl
        except ImportError:
            pl = None

        # If neither pandas nor polars is available, raise an ImportError
        if pd is None and pl is None:
            raise ImportError(
                "Generating a report with the `.report_as_html()` method requires either the "
                "Polars or the Pandas library."
            )

        # Prefer the use of the Polars library if available
        tbl_lib = pl if pl is not None else pd

        validation_info_dict = _validation_info_as_dict(validation_info=self.validation_info)

        # ------------------------------------------------
        # Process the `type` entry
        # ------------------------------------------------

        # Add the `type_upd` entry to the dictionary
        validation_info_dict["type_upd"] = _transform_assertion_str(
            assertion_str=validation_info_dict["assertion_type"]
        )

        # ------------------------------------------------
        # Process the `values` entry
        # ------------------------------------------------

        # Here, `values` will be transformed in ways particular to the assertion type (e.g.,
        # single values, ranges, sets, etc.)

        # Create a list to store the transformed values
        values_upd = []

        # Iterate over the values in the `values` entry
        values = validation_info_dict["values"]
        assertion_type = validation_info_dict["assertion_type"]
        inclusive = validation_info_dict["inclusive"]

        for i, value in enumerate(values):

            # If the assertion type is a comparison of one value then add the value as a string
            if assertion_type[i] in [
                "col_vals_gt",
                "col_vals_lt",
                "col_vals_eq",
                "col_vals_ne",
                "col_vals_ge",
                "col_vals_le",
            ]:
                values_upd.append(str(value))

            # If the assertion type is a comparison of values within or outside of a range, add
            # the appropriate brackets (inclusive or exclusive) to the values
            elif assertion_type[i] in ["col_vals_between", "col_vals_outside"]:
                left_bracket = "[" if inclusive[i][0] else "("
                right_bracket = "]" if inclusive[i][1] else ")"
                values_upd.append(f"{left_bracket}{value[0]}, {value[1]}{right_bracket}")

            # If the assertion type is a comparison of a set of values; strip the leading and
            # trailing square brackets and single quotes
            elif assertion_type[i] in ["col_vals_in_set", "col_vals_not_in_set"]:
                values_upd.append(str(value)[1:-1].replace("'", ""))

            # If the assertion type is not recognized, add the value as a string
            else:
                values_upd.append(str(value))

        # Add the `values_upd` entry to the dictionary
        validation_info_dict["values_upd"] = values_upd

        # ------------------------------------------------
        # Process `pass` and `fail` entries
        # ------------------------------------------------

        # Create a `pass` entry that concatenates the `n_passed` and `n_failed` entries (the length
        # of the `pass` entry should be equal to the length of the `n_passed` and `n_failed` entries)

        validation_info_dict["pass"] = _transform_passed_failed(
            n_passed_failed=validation_info_dict["n_passed"],
            f_passed_failed=validation_info_dict["f_passed"],
        )

        validation_info_dict["fail"] = _transform_passed_failed(
            n_passed_failed=validation_info_dict["n_failed"],
            f_passed_failed=validation_info_dict["f_failed"],
        )

        # ------------------------------------------------
        # Process `W`, `S`, `N` entries
        # ------------------------------------------------

        # Transform `warn`, `stop`, and `notify` to `w_upd`, `s_upd`, and `n_upd` entries
        validation_info_dict["w_upd"] = _transform_w_s_n(validation_info_dict["warn"], "#E5AB00")
        validation_info_dict["s_upd"] = _transform_w_s_n(validation_info_dict["stop"], "#CF142B")
        validation_info_dict["n_upd"] = _transform_w_s_n(validation_info_dict["notify"], "#439CFE")

        # Remove the `assertion_type` entry from the dictionary
        validation_info_dict.pop("assertion_type")

        # Remove the `values` entry from the dictionary
        validation_info_dict.pop("values")

        # Remove `n_passed`, `n_failed`, `f_passed`, and `f_failed` entries from the dictionary
        validation_info_dict.pop("n_passed")
        validation_info_dict.pop("n_failed")
        validation_info_dict.pop("f_passed")
        validation_info_dict.pop("f_failed")

        # Remove the `warn`, `stop`, and `notify` entries from the dictionary
        validation_info_dict.pop("warn")
        validation_info_dict.pop("stop")
        validation_info_dict.pop("notify")

        # Create a DataFrame from the validation information using the tbl_lib library
        # either Polars or Pandas
        df = tbl_lib.DataFrame(validation_info_dict)

        # Drop the following columns from the DataFrame
        df = df.drop(["inclusive", "na_pass", "label", "brief", "active", "all_passed"])

        # Return the DataFrame as a Great Tables table
        gt_tbl = (
            GT(df)
            .tab_header(title="Pointblank Validation")
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(style=style.css("height: 40px;"), locations=loc.body())
            .tab_style(
                style=style.text(weight="bold", color="#666666", size="13px"),
                locations=loc.body(columns="i"),
            )
            .tab_style(
                style=style.text(weight="bold", color="#666666"), locations=loc.column_labels()
            )
            .tab_style(
                style=style.text(size="28px", weight="bold", align="left", color="#444444"),
                locations=loc.title(),
            )
            .tab_style(
                style=style.text(
                    color="black", font=google_font(name="IBM Plex Mono"), size="11px"
                ),
                locations=loc.body(
                    columns=["type_upd", "column", "values_upd", "n", "pass", "fail"]
                ),
            )
            .tab_style(
                style=style.borders(sides="left", color="#E5E5E5", style="dashed"),
                locations=loc.body(columns=["column", "values_upd", "pass", "fail"]),
            )
            .tab_style(
                style=style.fill(color="#FCFCFC"),
                locations=loc.body(columns=["w_upd", "s_upd", "n_upd"]),
            )
            .tab_style(
                style=style.borders(sides="right", color="#D3D3D3"),
                locations=loc.body(columns="n_upd"),
            )
            .tab_style(
                style=style.borders(sides="left", color="#D3D3D3"),
                locations=loc.body(columns="w_upd"),
            )
            .cols_label(
                cases={
                    "i": "",
                    "type_upd": "STEP",
                    "column": "COLUMNS",
                    "values_upd": "VALUES",
                    "n": "UNITS",
                    "pass": "PASS",
                    "fail": "FAIL",
                    "w_upd": "W",
                    "s_upd": "S",
                    "n_upd": "N",
                }
            )
            .sub_missing(columns=["w_upd", "s_upd", "n_upd"], missing_text=html("&mdash;"))
            .cols_width(
                cases={
                    "i": "35px",
                    "type_upd": "190px",
                    "column": "120px",
                    "values_upd": "120px",
                    "n": "60px",
                    "pass": "60px",
                    "fail": "60px",
                    "w_upd": "30px",
                    "s_upd": "30px",
                    "n_upd": "30px",
                }
            )
            .cols_move_to_start(["i", "type_upd", "column", "values_upd", "n", "pass", "fail"])
            .fmt_markdown(columns=["pass", "fail"])
        )

        return gt_tbl

    def _add_validation(self, validation_info):
        """
        Add a validation to the list of validations.

        Parameters
        ----------
        validation_info : _ValidationInfo
            Information about the validation to add.
        """

        validation_info.i = len(self.validation_info) + 1

        self.validation_info.append(validation_info)

        return self

    def _get_validations(self):
        """
        Get the list of validations.

        Returns
        -------
        list[_ValidationInfo]
            The list of validations.
        """
        return self.validation_info

    def _clear_validations(self):
        """
        Clear the list of validations.
        """
        self.validation_info.clear()

        return self.validation_info

    def _get_validation_dict(self, i: int | list[int] | None, attr: str) -> dict[int, int]:
        """
        Utility function to get a dictionary of validation attributes for each validation step.

        Parameters
        ----------
        i : int | list[int] | None
            The validation step number(s) from which the attribute values are obtained.
            If `None`, all steps are included.
        attr : str
            The attribute name to retrieve from each validation step.

        Returns
        -------
        dict[int, int]
            A dictionary of the attribute values for each validation step.
        """
        if isinstance(i, int):
            i = [i]

        if i is None:
            return {validation.i: getattr(validation, attr) for validation in self.validation_info}

        return {
            validation.i: getattr(validation, attr)
            for validation in self.validation_info
            if validation.i in i
        }


def _check_boolean_input(param: bool, param_name: str):
    """
    Check that input value is a boolean.

    Parameters
    ----------
    param : bool
        The input value to check for a boolean value.
    param_name : str
        The name of the parameter being checked. This is used in the error message.

    Raises
    ------
    ValueError
        When `param=` is not a boolean value.
    """
    if not isinstance(param, bool):
        raise ValueError(f"`{param_name}=` must be a boolean value.")


def _check_column(column: str):
    """
    Check the input value of the `column=` parameter.

    Parameters
    ----------
    column : str
        The column to validate.

    Raises
    ------
    ValueError
        When `column` is not a string.
    """
    if not isinstance(column, str):
        raise ValueError("`column=` must be a string.")


def _check_value_float_int(value: float | int):
    """
    Check that input value of the `value=` parameter is a float or integer.

    Parameters
    ----------
    value : float | int
        The value to compare against in a validation.

    Raises
    ------
    ValueError
        When `value` is not a float or integer.
    """
    if not isinstance(value, (float, int)):
        raise ValueError("`value=` must be a float or integer.")


def _check_set_types(set: list[float | int | str]):
    """
    Check that input value of the `set=` parameter is a list of floats, integers, or strings.

    Parameters
    ----------
    set : list[float | int]
        The set of values to compare against in a validation.

    Raises
    ------
    ValueError
        When `set` is not a list of floats or integers.
    """
    if not all(isinstance(value, (float, int, str)) for value in set):
        raise ValueError("`set=` must be a list of floats, integers, or strings.")


def _check_thresholds(thresholds: int | float | tuple | dict | Thresholds | None):
    """
    Check that input value of the `thresholds=` parameter is a valid threshold.

    Parameters
    ----------
    thresholds : int | float | tuple | dict | Thresholds | None
        The threshold value or values.

    Raises
    ------
    ValueError
        When `thresholds` is not a valid threshold.
    """

    # Raise a ValueError if the thresholds argument is not valid (also accept None)
    if thresholds is not None and not isinstance(thresholds, (int, float, tuple, dict, Thresholds)):
        raise ValueError("The thresholds argument is not valid.")


def _validation_info_as_dict(validation_info: _ValidationInfo) -> dict:
    """
    Convert a `_ValidationInfo` object to a dictionary.

    Parameters
    ----------
    validation_info : _ValidationInfo
        The `_ValidationInfo` object to convert to a dictionary.

    Returns
    -------
    dict
        A dictionary representing the `_ValidationInfo` object.
    """

    # Define the fields to include in the validation information
    validation_info_fields = [
        "i",
        "assertion_type",
        "column",
        "values",
        "inclusive",
        "na_pass",
        "label",
        "brief",
        "active",
        "all_passed",
        "n",
        "n_passed",
        "n_failed",
        "f_passed",
        "f_failed",
        "warn",
        "stop",
        "notify",
    ]

    # Filter the validation information to include only the selected fields
    validation_info_filtered = [
        {field: getattr(validation, field) for field in validation_info_fields}
        for validation in validation_info
    ]

    # Transform the validation information into a dictionary of lists so that it
    # can be used to create a DataFrame
    validation_info_dict = {field: [] for field in validation_info_fields}

    for validation in validation_info_filtered:
        for field in validation_info_fields:
            validation_info_dict[field].append(validation[field])

    return validation_info_dict


def _transform_w_s_n(values, color):
    return [
        (
            "&mdash;"
            if value is None
            else (
                f'<span style="color: {color};">&#9679;</span>'
                if value is True
                else f'<span style="color: {color};">&cir;</span>' if value is False else value
            )
        )
        for value in values
    ]


def _get_assertion_icon(icon: list[str] | None, length_val: int = 30) -> list[str]:

    # If icon is None, return an empty list
    if icon is None:
        return []

    # For each icon, get the assertion icon SVG test from SVG_ICONS_FOR_ASSERTION_TYPES dictionary
    icon_svg = [SVG_ICONS_FOR_ASSERTION_TYPES.get(icon) for icon in icon]

    # Replace the width and height in the SVG string
    for i in range(len(icon_svg)):
        icon_svg[i] = _replace_svg_dimensions(icon_svg[i], height_width=length_val)

    return icon_svg


def _replace_svg_dimensions(svg: list[str], height_width: int | float) -> list[str]:

    svg = re.sub(r'width="[0-9]*?px', f'width="{height_width}px', svg)
    svg = re.sub(r'height="[0-9]*?px', f'height="{height_width}px', svg)

    return svg


def _transform_passed_failed(n_passed_failed: list[int], f_passed_failed: list[float]) -> list[str]:

    passed_failed = [
        f"{n_passed_failed[i]}<br />{f_passed_failed[i]}" for i in range(len(n_passed_failed))
    ]

    return passed_failed


def _transform_assertion_str(assertion_str: list[str]) -> list[str]:

    # Get the SVG icons for the assertion types
    svg_icon = _get_assertion_icon(icon=assertion_str)

    # Append `()` to the `assertion_str`
    assertion_str = [x + "()" for x in assertion_str]

    # Obtain the number of characters contained in the assertion
    # string; this is important for sizing components appropriately
    assertion_type_nchar = [len(x) for x in assertion_str]

    # Declare the text size based on the length of `assertion_str`
    text_size = [10 if nchar + 2 >= 20 else 11 for nchar in assertion_type_nchar]

    # Create the assertion type update using a list comprehension
    type_upd = [
        f"""
        <div style="margin:0;padding:0;display:inline-block;height:30px;vertical-align:middle;">
        <!--?xml version="1.0" encoding="UTF-8"?-->{svg}
        </div>
        <span style="font-family: 'IBM Plex Mono', monospace, courier; color: black; font-size:{size}px;"> {assertion}</span>
        """
        for assertion, svg, size in zip(assertion_str, svg_icon, text_size)
    ]

    return type_upd
