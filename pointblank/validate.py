from __future__ import annotations

from dataclasses import dataclass, field

from narwhals.typing import FrameT


@dataclass
class ValidationInfo:
    """
    Information about a validation to be performed on a table and the results of the interrogation.
    """

    i: int | None = None
    i_o: int | None = None
    step_id: str | None = None
    sha1: str | None = None
    assertion_type: str | None = None
    column: str | None = None
    values: list | None = None
    na_pass: bool | None = None
    label: str | None = None
    brief: str | None = None
    active: bool | None = None
    all_passed: bool | None = None
    n: int | None = None
    n_passed: int | None = None
    n_failed: int | None = None
    f_passed: int | None = None
    f_failed: int | None = None
    warn: bool | None = None
    notify: bool | None = None
    stop: bool | None = None
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

    def clear_validations(self):
        """
        Clear the list of validations.
        """
        self.validation_info.clear()

    def col_vals_gt(self, column, value, threshold: int | float | None = None):
        """
        Add a validation to check if the values in a column are greater than a threshold.

        Parameters
        ----------
        column : str
            The column to validate.
        value : int | float
            The value to compare against.
        threshold : int | float
            The threshold value.
        """

        val_info = ValidationInfo(
            assertion_type="col_vals_gt",
            column=column,
            values=value,
            active=True,
            all_passed=False,
        )

        self.add_validation(val_info)

        return self

    def interrogate(self):
        """
        Evaluate each validation against the table and store the results.
        """
        from pointblank.test import Test

        for validation in self.validation_info:
            type = validation.assertion_type
            column = validation.column
            values = validation.values

            if type == "col_vals_gt":
                result_tf = Test.col_vals_gt(df=self.data, column=column, value=values)
                validation.all_passed = result_tf

        return self
