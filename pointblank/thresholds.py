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

