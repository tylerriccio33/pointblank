from __future__ import annotations

from typing import Callable
from pointblank.thresholds import Thresholds


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


def _check_pre(pre: Callable | None):
    """
    Check that input value of the `pre=` parameter is a callable function.

    Parameters
    ----------
    pre : Callable | None
        The pre-processing function to apply to the table.

    Raises
    ------
    ValueError
        When `pre` is not a callable function.
    """
    if pre is not None and not isinstance(pre, Callable):
        raise ValueError("`pre=` must be a callable function.")


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

    if thresholds is None or isinstance(thresholds, Thresholds):
        return

    if isinstance(thresholds, (int, float)):
        if thresholds < 0:
            raise ValueError(
                "If an int or float is supplied to `thresholds=` it must be a "
                "non-negative value."
            )

    if isinstance(thresholds, tuple):
        if len(thresholds) > 3:
            raise ValueError(
                "If a tuple is supplied to `thresholds=` it must have at most three elements."
            )
        if not all(isinstance(threshold, (int, float)) for threshold in thresholds):
            raise ValueError(
                "If a tuple is supplied to `thresholds=` all elements must be integers or floats."
            )
        if any(threshold < 0 for threshold in thresholds):
            raise ValueError(
                "If a tuple is supplied to `thresholds=` all elements must be non-negative."
            )

    if isinstance(thresholds, dict):

        # Check keys for invalid entries and raise a ValueError if any are found
        invalid_keys = set(thresholds.keys()) - {"warn_at", "stop_at", "notify_at"}

        if invalid_keys:
            raise ValueError(f"Invalid keys in the thresholds dictionary: {invalid_keys}")

        # Get values as a list and raise a ValueError for any non-integer or non-float values
        values = list(thresholds.values())

        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError(
                "If a dict is supplied to `thresholds=` all values must be integers or floats."
            )

        # Raise a ValueError if any values are negative
        if any(value < 0 for value in values):
            raise ValueError(
                "If a dict is supplied to `thresholds=` all values must be non-negative."
            )

    # Raise a ValueError if the thresholds argument is not valid (also accept None)
    if thresholds is not None and not isinstance(thresholds, (int, float, tuple, dict, Thresholds)):
        raise ValueError("The thresholds argument is not valid.")
