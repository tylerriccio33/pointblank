from __future__ import annotations

import datetime
import json

from dataclasses import dataclass, field

import narwhals as nw
from narwhals.typing import FrameT
from great_tables import GT, md, html, loc, style, google_font

from pointblank._constants import (
    TYPE_METHOD_MAP,
    COMPATIBLE_TYPES,
    COMPARE_TYPE_MAP,
    VALIDATION_REPORT_FIELDS,
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
class ValidationInfo:
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
        Whether the table has undergone validation. This may later be the table itself augmented
        with a column that indicates the result of the validation (but only for column-value based
        validations).
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
    """
    A class to represent a table validation.
    """

    data: FrameT
    validation_info: list[ValidationInfo] = field(default_factory=list)

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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

        val_info = ValidationInfo(
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

            type = validation.assertion_type
            column = validation.column
            value = validation.values
            inclusive = validation.inclusive
            na_pass = validation.na_pass
            threshold = validation.thresholds

            comparison = TYPE_METHOD_MAP[type]
            compare_type = COMPARE_TYPE_MAP[comparison]
            compatible_types = COMPATIBLE_TYPES.get(comparison, [])

            validation.n = NumberOfTestUnits(df=df, column=column).get_test_units()

            if compare_type == "COMPARE_ONE":

                results_list = ColValsCompareOne(
                    df=df,
                    column=column,
                    value=value,
                    na_pass=na_pass,
                    threshold=threshold,
                    comparison=comparison,
                    allowed_types=compatible_types,
                ).get_test_results()

            if compare_type == "COMPARE_TWO":

                results_list = ColValsCompareTwo(
                    df=df,
                    column=column,
                    value1=value[0],
                    value2=value[1],
                    inclusive=inclusive,
                    na_pass=na_pass,
                    threshold=threshold,
                    comparison=comparison,
                    allowed_types=compatible_types,
                ).get_test_results()

            if compare_type == "COMPARE_SET":

                inside = True if comparison == "in_set" else False

                results_list = ColValsCompareSet(
                    df=df,
                    column=column,
                    values=value,
                    threshold=threshold,
                    inside=inside,
                    allowed_types=compatible_types,
                ).get_test_results()

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

            # Set the table as checked
            validation.tbl_checked = True

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

        return {
            validation.i: validation.n_passed
            for validation in self.validation_info
            if validation.i in i
        }

    def get_report(self):

        validation_info_list = self.validation_info

        report = []

        for validation_info in validation_info_list:

            report.append(
                {
                    "i": validation_info.i,
                    "assertion_type": validation_info.assertion_type,
                    "column": validation_info.column,
                    "values": validation_info.values,
                    "na_pass": validation_info.na_pass,
                    "thresholds": validation_info.thresholds,
                    "label": validation_info.label,
                    "brief": validation_info.brief,
                    "active": validation_info.active,
                    "all_passed": validation_info.all_passed,
                    "n": validation_info.n,
                    "n_passed": validation_info.n_passed,
                    "n_failed": validation_info.n_failed,
                    "f_passed": validation_info.f_passed,
                    "f_failed": validation_info.f_failed,
                    "warn": validation_info.warn,
                    "stop": validation_info.stop,
                    "notify": validation_info.notify,
                    "row_sample": validation_info.row_sample,
                    "tbl_checked": validation_info.tbl_checked,
                    "time_processed": validation_info.time_processed,
                    "proc_duration_s": validation_info.proc_duration_s,
                }
            )

        return json.dumps(report, indent=4, default=str)

    def _add_validation(self, validation_info):
        """
        Add a validation to the list of validations.

        Parameters
        ----------
        validation_info : ValidationInfo
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
        list[ValidationInfo]
            The list of validations.
        """
        return self.validation_info

    def _clear_validations(self):
        """
        Clear the list of validations.
        """
        self.validation_info.clear()

        return self.validation_info
