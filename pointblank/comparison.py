from __future__ import annotations
from dataclasses import dataclass

import narwhals as nw


@dataclass
class Comparator:
    x: float | int | list[float | int] | nw.DataFrame
    column: str = None
    compare: float | int | list[float | int] = None
    low: float | int | list[float | int] = None
    high: float | int | list[float | int] = None

    # If x or compare is a scalar, convert to a list

    def __post_init__(self):
        if not isinstance(self.x, list):
            self.x = [self.x]

        if self.compare is not None:
            self.compare = self._ensure_list(self.compare, len(self.x), "compare")

        if self.low is not None:
            self.low = self._ensure_list(self.low, len(self.x), "low")

        if self.high is not None:
            self.high = self._ensure_list(self.high, len(self.x), "high")

    def _ensure_list(self, value, length=None, name=None):
        if not isinstance(value, list):
            value = [value] * (length if length is not None else 1)
        elif length is not None and len(value) != length:
            raise ValueError(f"Length of `x` and `{name}` must be the same.")

        return value

    def gt(self) -> list[bool]:
        return [i > j for i, j in zip(self.x, self.compare)]

    def lt(self) -> list[bool]:
        return [i < j for i, j in zip(self.x, self.compare)]

    def eq(self) -> list[bool]:
        return [i == j for i, j in zip(self.x, self.compare)]

    def ne(self) -> list[bool]:
        return [i != j for i, j in zip(self.x, self.compare)]

    def ge(self) -> list[bool]:
        return [i >= j for i, j in zip(self.x, self.compare)]

    def le(self) -> list[bool]:
        return [i <= j for i, j in zip(self.x, self.compare)]

    def between(self) -> list[bool]:
        return [i > j and i < k for i, j, k in zip(self.x, self.low, self.high)]

    def outside(self) -> list[bool]:
        return [i < j or i > k for i, j, k in zip(self.x, self.low, self.high)]

    def isin(self) -> list[bool]:
        return [i in self.compare for i in self.x]

    def notin(self) -> list[bool]:
        return [i not in self.compare for i in self.x]

    def isnull(self) -> list[bool]:
        return [i is None for i in self.x]

    def notnull(self) -> list[bool]:
        return [i is not None for i in self.x]
