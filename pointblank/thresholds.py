from __future__ import annotations

from typing import Callable
from dataclasses import dataclass, field

__all__ = ["Thresholds", "Actions"]


@dataclass
class Thresholds:
    """
    Definition of threshold values.

    Thresholds are used to set limits on the number of failing test units at different levels. The
    levels are 'warning', 'error', and 'critical'. These levels correspond to different levels of
    severity when a threshold is reached. The threshold values can be set as absolute counts or as
    fractions of the total number of test units. When a threshold is reached, an action can be taken
    (e.g., displaying a message or calling a function) if there is an associated action defined for
    that level (defined through the [`Actions`](`pointblank.Actions`) class).

    Parameters
    ----------
    warning
        The threshold for the 'warning' level. This can be an absolute count or a fraction of the
        total. Using `True` will set this threshold value to `1`.
    error
        The threshold for the 'error' level. This can be an absolute count or a fraction of the
        total. Using `True` will set this threshold value to `1`.
    critical
        The threshold for the 'critical' level. This can be an absolute count or a fraction of the
        total. Using `True` will set this threshold value to `1`.

    Returns
    -------
    Thresholds
        A `Thresholds` object. This can be used when using the [`Validate`](`pointblank.Validate`)
        class (to set thresholds globally) or when defining validation steps like
        [`col_vals_gt()`](`pointblank.Validate.col_vals_gt`) (so that threshold values are scoped to
        individual validation steps, overriding any global thresholds).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer=False)
    ```
    In a data validation workflow, you can set thresholds for the number of failing test units at
    different levels. For example, you can set a threshold for the 'warning' level when the number
    of failing test units exceeds 10% of the total number of test units:

    ```{python}
    thresholds_1 = pb.Thresholds(warning=0.1)
    ```

    You can also set thresholds for the 'error' and 'critical' levels:

    ```{python}
    thresholds_2 = pb.Thresholds(warning=0.1, error=0.2, critical=0.05)
    ```

    Thresholds can also be set as absolute counts. Here's an example where the 'warning' level is
    set to `5` failing test units:

    ```{python}
    thresholds_3 = pb.Thresholds(warning=5)
    ```

    The `thresholds` object can be used to set global thresholds for all validation steps. Or, you
    can set thresholds for individual validation steps, which will override the global thresholds.
    Here's a data validation workflow example where we set global thresholds and then override with
    different thresholds at the [`col_vals_gt()`](`pointblank.Validate.col_vals_gt`) step:

    ```{python}
    validation = (
        pb.Validate(
            data=pb.load_dataset(dataset="small_table"),
            label="Example Validation",
            thresholds=pb.Thresholds(warning=0.1, error=0.2, critical=0.3)
        )
        .col_vals_not_null(columns=["c", "d"])
        .col_vals_gt(columns="a", value=3, thresholds=pb.Thresholds(warning=5))
        .interrogate()
    )

    validation
    ```

    As can be seen, the last step ([`col_vals_gt()`](`pointblank.Validate.col_vals_gt`)) has its own
    thresholds, which override the global thresholds set at the beginning of the validation workflow
    (in the [`Validate`](`pointblank.Validate`) class).
    """

    warning: int | float | bool | None = None
    error: int | float | bool | None = None
    critical: int | float | bool | None = None

    warning_fraction: float | None = field(default=None, init=False)
    warning_count: int | None = field(default=None, init=False)
    error_fraction: float | None = field(default=None, init=False)
    error_count: int | None = field(default=None, init=False)
    critical_fraction: float | None = field(default=None, init=False)
    critical_count: int | None = field(default=None, init=False)

    def __post_init__(self):
        self._process_threshold("warning", "warning")
        self._process_threshold("error", "error")
        self._process_threshold("critical", "critical")

    def _process_threshold(self, attribute_name, base_name):
        value = getattr(self, attribute_name)
        if value is not None:
            if value == 0:
                setattr(self, f"{base_name}_fraction", 0)
                setattr(self, f"{base_name}_count", 0)
            elif 0 < value < 1:
                setattr(self, f"{base_name}_fraction", value)
            elif value >= 1:
                setattr(self, f"{base_name}_count", round(value))
            elif value < 0:
                raise ValueError(f"Negative values are not allowed for `{attribute_name}`.")

    def __repr__(self) -> str:
        return f"Thresholds(warning={self.warning}, error={self.error}, critical={self.critical})"

    def __str__(self) -> str:
        return self.__repr__()

    def _get_threshold_value(self, level: str) -> float | int | None:

        # The threshold for a given level (warning, error, critical) is either:
        # 1. a fraction
        # 2. an absolute count
        # 3. zero
        # 4. None

        # If the threshold is zero, return 0
        if getattr(self, f"{level}_count") == 0:
            return 0

        # If the threshold is a fraction, return the fraction
        if getattr(self, f"{level}_fraction") is not None:
            return getattr(self, f"{level}_fraction")

        # If the threshold is an absolute count, return the count
        if getattr(self, f"{level}_count") is not None:
            return getattr(self, f"{level}_count")

        # The final case is where the threshold is None, so None is returned
        return None

    def _threshold_result(
        self, fraction_failing: float, test_units: int, level: str
    ) -> bool | None:
        """
        Determine if the number of failing test units is below the threshold.

        Parameters
        ----------
        failing_test_units
            The number of failing test units.
        level
            The threshold level to check.

        Returns
        -------
            `True` when test units pass below the threshold level for failing test units, `False`
            when the opposite. If the threshold is `None` (i.e., no threshold is set), `None` is
            returned.
        """

        threshold_value = self._get_threshold_value(level=level)

        if threshold_value is None:
            return None

        if threshold_value == 0:
            return True

        # The threshold value might be an absolute count, but we need to convert
        # it to a fractional value
        if isinstance(threshold_value, int):
            threshold_value = _convert_abs_count_to_fraction(
                value=threshold_value, test_units=test_units
            )

        return fraction_failing >= threshold_value


def _convert_abs_count_to_fraction(value: int | None, test_units: int) -> float:

    # Using a integer value signifying the total number of 'test units' (in the
    # context of a validation), we convert an integer count (absolute) threshold
    # value to a fractional threshold value

    if test_units == 0:
        raise ValueError("The total number of test units must be greater than zero.")

    if test_units < 0:
        raise ValueError("The total number of test units must be a positive integer.")

    if value is None:
        return None

    if value < 0:
        raise ValueError("Negative values are not allowed for threshold counts.")

    if value == 0:
        return 0.0

    return float(round(value) / test_units)


def _normalize_thresholds_creation(
    thresholds: int | float | tuple | dict | Thresholds | None,
) -> Thresholds:
    """
    Normalize the thresholds argument to a Thresholds object.

    Parameters
    ----------
    thresholds
        The value or values to use for the thresholds.

    Returns
    -------
    Thresholds
        The normalized Thresholds object.
    """

    if thresholds is None:
        thresholds = Thresholds()

    elif isinstance(thresholds, (int, float)):
        thresholds = Thresholds(warning=thresholds)

    elif isinstance(thresholds, tuple):

        # The tuple should have 1-3 elements
        if len(thresholds) == 1:
            thresholds = Thresholds(warning=thresholds[0])
        elif len(thresholds) == 2:
            thresholds = Thresholds(warning=thresholds[0], error=thresholds[1])
        elif len(thresholds) == 3:
            thresholds = Thresholds(
                warning=thresholds[0], error=thresholds[1], critical=thresholds[2]
            )
        else:
            raise ValueError("The tuple should have 1-3 elements.")

    elif isinstance(thresholds, dict):

        # The dictionary should have keys for "warning", "error", and "critical"; it can omit
        # any of these keys

        # Check keys for invalid entries and raise a ValueError if any are found
        invalid_keys = set(thresholds.keys()) - {"warning", "error", "critical"}

        if invalid_keys:
            raise ValueError(f"Invalid keys in the thresholds dictionary: {invalid_keys}")

        thresholds = Thresholds(**thresholds)

    elif isinstance(thresholds, Thresholds):
        pass

    else:
        raise ValueError("The thresholds argument is not valid.")

    return thresholds


def _threshold_check(failing_test_units: int, threshold: int | None) -> bool:
    """
    Determine if the number of failing test units is below the threshold.

    Parameters
    ----------
    failing_test_units
        The number of failing test units.
    threshold
        The maximum number of failing test units to allow.

    Returns
    -------
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    if threshold is None:
        return False

    return failing_test_units < threshold


@dataclass
class Actions:
    """
    Definition of action values.

    Actions complement threshold values by defining what action should be taken when a threshold
    level is reached. The action can be a string or a `Callable`. When a string is used, it is
    interpreted as a message to be displayed. When a `Callable` is used, it will be invoked at
    interrogation time if the threshold level is met or exceeded.

    There are three threshold levels: 'warning', 'error', and 'critical'. These levels correspond
    to different levels of severity when a threshold is reached. Those thresholds can be defined
    using the [`Thresholds`](`pointblank.Thresholds`) class or various shorthand forms. Actions
    don't have to be defined for all threshold levels; if an action is not defined for a level in
    exceedence, no action will be taken.

    Parameters
    ----------
    warning
        A string, `Callable`, or list of `Callable`/string values for the 'warning' level. Using
        `None` means no action should be performed at the 'warning' level.
    error
        A string, `Callable`, or list of `Callable`/string values for the 'error' level. Using
        `None` means no action should be performed at the 'error' level.
    critical
        A string, `Callable`, or list of `Callable`/string values for the 'critical' level. Using
        `None` means no action should be performed at the 'critical' level.

    Returns
    -------
    Actions
        An `Actions` object. This can be used when using the [`Validate`](`pointblank.Validate`)
        class (to set actions for meeting different threshold levels globally) or when defining
        validation steps like [`col_vals_gt()`](`pointblank.Validate.col_vals_gt`) (so that actions
        are scoped to individual validation steps, overriding any globally set actions).

    Types of Actions
    ----------------
    Actions can be defined in different ways:

    1. **String**: A message to be displayed when the threshold level is met or exceeded.
    2. **Callable**: A function that is called when the threshold level is met or exceeded.
    3. **List of Strings/Callables**: Multiple messages or functions to be called when the threshold
       level is met or exceeded.

    The actions are executed at interrogation time when the threshold level assigned to the action
    is exceeded by the number or proportion of failing test units. When providing a string, it will
    simply be printed to the console. A callable will also be executed at the time of interrogation.
    If providing a list of strings or callables, each item in the list will be executed in order.
    Such a list can contain a mix of strings and callables.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer=False)
    ```

    Let's define both threshold values and actions for a data validation workflow. We'll set these
    thresholds and actions globally for all validation steps. In this specific example, the only
    actions we'll define are for the 'critical' level:

    ```{python}
    import pointblank as pb

    validation = (
        pb.Validate(
            data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb"),
            thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
            actions=pb.Actions(critical="Major data quality issue found."),
        )
        .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}\d{3}")
        .col_vals_gt(columns="item_revenue", value=0.05)
        .col_vals_gt(columns="session_duration", value=15)
        .interrogate()
    )

    validation
    ```

    Because we set the 'critical' action to display `"Major data quality issue found."` in the
    console, this message will be displayed if the number of failing test units exceeds the
    'critical' threshold (set to 15% of the total number of test units). In step 3 of the validation
    workflow, the 'critical' threshold is exceeded, so the message is displayed in the console.

    Actions can be defined locally for individual validation steps, which will override any global
    actions set at the beginning of the validation workflow. Here's a variation of the above example
    where we set global threshold values but assign an action only for an individual validation
    step:

    ```{python}
    def dq_issue():
        from datetime import datetime

        print(f"Data quality issue found ({datetime.now()}).")

    validation = (
        pb.Validate(
            data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb"),
            thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
        )
        .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}\d{3}")
        .col_vals_gt(columns="item_revenue", value=0.05)
        .col_vals_gt(
            columns="session_duration",
            value=15,
            actions=pb.Actions(warning=dq_issue),
        )
        .interrogate()
    )

    validation
    ```

    In this case, the 'warning' action is set to call the `dq_issue()` function with the argument
    `"session duration column"`. This action is only executed when the 'warning' threshold is
    exceeded in the 'session_duration' column. Because all three thresholds are exceeded in step
    3, the 'warning' action of executing the function occurs (resulting in a message being printed
    to the console).
    """

    warning: str | Callable | list[str | Callable] | None = None
    error: str | Callable | list[str | Callable] | None = None
    critical: str | Callable | list[str | Callable] | None = None

    def __post_init__(self):
        self.warning = self._ensure_list(self.warning)
        self.error = self._ensure_list(self.error)
        self.critical = self._ensure_list(self.critical)

    def _ensure_list(
        self, value: str | Callable | list[str | Callable] | None
    ) -> list[str | Callable]:
        if value is None:
            return None
        if not isinstance(value, list):
            return [value]
        return value

    def __repr__(self) -> str:
        return f"Actions(warning={self.warning}, error={self.error}, critical={self.critical})"

    def __str__(self) -> str:
        return self.__repr__()

    def _get_action(self, level: str) -> list[str | Callable]:
        return getattr(self, level)
