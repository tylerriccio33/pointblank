from __future__ import annotations

import datetime

from dataclasses import dataclass, field

from narwhals.typing import FrameT

from pointblank._constants import TYPE_METHOD_MAP, COMPATIBLE_TYPES, COMPARE_TYPE_MAP
from pointblank._comparison import ColValsCompareOne, ColValsCompareTwo
from pointblank._utils import _get_assertion_type_from_fname
from pointblank.thresholds import (
    Thresholds,
    _normalize_thresholds_creation,
    _convert_abs_count_to_fraction,
)


@dataclass
class ValidationInfo:
    """
    Information about a validation to be performed on a table and the results of the interrogation.
    """

    # Validation plan
    i: int | None = None
    i_o: int | None = None
    step_id: str | None = None
    sha1: str | None = None
    assertion_type: str | None = None
    column: str | None = None
    values: any | list[any] | tuple | None = None
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

    def add_validation(self, validation_info):
        """
        Add a validation to the list of validations.

        Parameters
        ----------
        validation_info : ValidationInfo
            Information about the validation to add.
        """
        self.validation_info.append(validation_info)

    def get_validations(self):
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

    def col_vals_gt(
        self,
        column: str,
        value: float | int,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values are greater than a fixed value.

        Parameters
        ----------
        column : str
            The column to validate.
        value : int | float
            The value to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_lt(
        self,
        column: str,
        value: float | int,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values are less than a fixed value.

        Parameters
        ----------
        column : str
            The column to validate.
        value : int | float
            The value to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_eq(
        self,
        column: str,
        value: float | int,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values are equal to a fixed value.

        Parameters
        ----------
        column : str
            The column to validate.
        value : int | float
            The value to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_ne(
        self,
        column: str,
        value: float | int,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values are not equal to a fixed value.

        Parameters
        ----------
        column : str
            The column to validate.
        value : int | float
            The value to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_ge(
        self,
        column: str,
        value: float | int,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values greater than or equal to a fixed value.

        Parameters
        ----------
        column : str
            The column to validate.
        value : int | float
            The value to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_le(
        self,
        column: str,
        value: float | int,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values less than or equal to a fixed value.

        Parameters
        ----------
        column : str
            The column to validate.
        value : int | float
            The value to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_between(
        self,
        column: str,
        left: float | int,
        right: float | int,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values less than or equal to a fixed value.

        Parameters
        ----------
        column : str
            The column to validate.
        value : int | float
            The value to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        value = (left, right)

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_outside(
        self,
        column: str,
        left: float | int,
        right: float | int,
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values are outside of a range.

        Parameters
        ----------
        column : str
            The column to validate.
        left : int | float
            The lower bound of the range.
        right : int | float
            The upper bound of the range.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        value = (left, right)

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=value,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_in_set(
        self,
        column: str,
        values: list[float | int],
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values are in a set of values.

        Parameters
        ----------
        column : str
            The column to validate.
        values : list[int | float]
            The values to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=values,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def col_vals_not_in_set(
        self,
        column: str,
        values: list[float | int],
        thresholds: int | float | tuple | dict | Thresholds = None,
        active: bool = True,
    ):
        """
        Add a validation to check if column values are not in a set of values.

        Parameters
        ----------
        column : str
            The column to validate.
        values : list[int | float]
            The values to compare against.
        thresholds : int | float | tuple | dict| Thresholds, optional
            The threshold value or values.
        active : bool, optional
            Whether the validation is active.
        """

        assertion_type = _get_assertion_type_from_fname()

        val_info = ValidationInfo(
            assertion_type=assertion_type,
            column=column,
            values=values,
            thresholds=_normalize_thresholds_creation(thresholds),
            active=active,
        )

        self.add_validation(val_info)

        return self

    def interrogate(self):
        """
        Evaluate each validation against the table and store the results.
        """

        df = self.data

        for validation in self.validation_info:

            start_time = datetime.datetime.now(datetime.timezone.utc)

            type = validation.assertion_type
            column = validation.column
            value = validation.values
            threshold = validation.thresholds

            comparison = TYPE_METHOD_MAP[type]
            compare_type = COMPARE_TYPE_MAP[comparison]
            compatible_types = COMPATIBLE_TYPES.get(comparison, [])

            if compare_type == "COMPARE_ONE":

                results_list = ColValsCompareOne(
                    df=df,
                    column=column,
                    value=value,
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
                    threshold=threshold,
                    comparison=comparison,
                    allowed_types=compatible_types,
                ).get_test_results()

            if compare_type == "COMPARE_SET":

                results_list = ColValsCompareOne(
                    df=df,
                    column=column,
                    value=value,
                    threshold=threshold,
                    comparison=comparison,
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
