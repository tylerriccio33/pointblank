from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Thresholds"]


@dataclass
class Thresholds:
    """
    Definition of threshold values.

    Parameters
    ----------
    warn_at
        The threshold for the 'warn' level. This can be an absolute count or a fraction of the
        total. Using `True` will set this threshold to 1.
    stop_at
        The threshold for the 'stop' level. This can be an absolute count or a fraction of the
        total. Using `True` will set this threshold to 1.
    notify_at
        The threshold for the 'notify' level. This can be an absolute count or a fraction of the
        total. Using `True` will set this threshold to 1.

    Returns
    -------
    Thresholds
        A Thresholds object. This can be used when using the `Validate` class (to set thresholds
        globally) or when defining validation steps through `Validate`'s methods (so that threshold
        values are scoped to individual validation steps, overriding any global thresholds).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer=False)
    ```
    In a data validation workflow, you can set thresholds for the number of failing test units at
    different levels. For example, you can set a threshold to warn when the number of failing test
    units exceeds 10% of the total number of test units:

    ```{python}
    thresholds = pb.Thresholds(warn_at=0.1)

    thresholds
    ```

    You can also set thresholds for the 'stop' and 'notify' levels:

    ```{python}
    thresholds = pb.Thresholds(warn_at=0.1, stop_at=0.2, notify_at=0.05)

    thresholds
    ```

    Thresholds can also be set as absolute counts. Here's an example where the 'warn' level is set
    to `5` failing test units:

    ```{python}
    thresholds = pb.Thresholds(warn_at=5)

    thresholds
    ```

    The `Thresholds` object can be used to set global thresholds for all validation steps. Or, you
    can set thresholds for individual validation steps, which will override the global thresholds.
    Here's a data validation workflow example where we set global thresholds and then override with
    different thresholds at the `col_vals_gt()` step:

    ```{python}
    validation = (
        pb.Validate(
            data=pb.load_dataset(dataset="small_table"),
            label="Example Validation",
            thresholds=pb.Thresholds(warn_at=0.1, stop_at=0.2, notify_at=0.3)
        )
        .col_vals_not_null(columns=["c", "d"])
        .col_vals_gt(columns="a", value=3, thresholds=pb.Thresholds(warn_at=5))
        .interrogate()
    )

    validation
    ```

    As can be seen, the last step (`col_vals_gt()`) has its own thresholds, which override the
    global thresholds set at the beginning of the validation workflow (in the `Validate` class).
    """

    warn_at: int | float | bool | None = None
    stop_at: int | float | bool | None = None
    notify_at: int | float | bool | None = None

    warn_fraction: float | None = field(default=None, init=False)
    warn_count: int | None = field(default=None, init=False)
    stop_fraction: float | None = field(default=None, init=False)
    stop_count: int | None = field(default=None, init=False)
    notify_fraction: float | None = field(default=None, init=False)
    notify_count: int | None = field(default=None, init=False)

    def __post_init__(self):
        self._process_threshold("warn_at", "warn")
        self._process_threshold("stop_at", "stop")
        self._process_threshold("notify_at", "notify")

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
        return f"Thresholds(warn_at={self.warn_at}, stop_at={self.stop_at}, notify_at={self.notify_at})"

    def __str__(self) -> str:
        return self.__repr__()

    def _get_threshold_value(self, level: str) -> float | int | None:

        # The threshold for a given level (warn, stop, notify) is either:
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
        thresholds = Thresholds(warn_at=thresholds)

    elif isinstance(thresholds, tuple):

        # The tuple should have 1-3 elements
        if len(thresholds) == 1:
            thresholds = Thresholds(warn_at=thresholds[0])
        elif len(thresholds) == 2:
            thresholds = Thresholds(warn_at=thresholds[0], stop_at=thresholds[1])
        elif len(thresholds) == 3:
            thresholds = Thresholds(
                warn_at=thresholds[0], stop_at=thresholds[1], notify_at=thresholds[2]
            )
        else:
            raise ValueError("The tuple should have 1-3 elements.")

    elif isinstance(thresholds, dict):

        # The dictionary should have keys for "warn_at", "stop_at", and "notify_at"; it can omit
        # any of these keys

        # Check keys for invalid entries and raise a ValueError if any are found
        invalid_keys = set(thresholds.keys()) - {"warn_at", "stop_at", "notify_at"}

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
