from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Thresholds:
    """
    A class to represent thresholds for a validation.
    """

    warn_at: int | float | None = None
    stop_at: int | float | None = None
    notify_at: int | float | None = None

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

        # If the threshold is a fraction, return the fraction
        if getattr(self, f"{level}_fraction") is not None:
            return getattr(self, f"{level}_fraction")

        # If the threshold is an absolute count, return the count
        if getattr(self, f"{level}_count") is not None:
            return getattr(self, f"{level}_count")

        # If the threshold is zero, return 0
        if getattr(self, f"{level}_count") == 0:
            return 0

        # The final case is where the threshold is None, so None is returned
        return None


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
