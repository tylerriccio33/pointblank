from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.series import Series


class _TypeMap(Enum):  # ! ordered
    STRUCT = ("struct",)
    NUMERIC = ("int", "float")
    STRING = ("string", "categorical")
    DATE = ("date",)
    BOOL = ("bool",)

    @classmethod
    def fetch_prof_map(cls) -> dict[_TypeMap, type[ColumnProfile]]:
        default = defaultdict(lambda: ColumnProfile)
        implemented_dict: dict[_TypeMap, type[ColumnProfile]] = {
            cls.BOOL: _BoolProfile,
            cls.NUMERIC: _NumericProfile,
            cls.STRING: _StringProfile,
            cls.DATE: _DateProfile,
        }
        return default | implemented_dict

    @classmethod
    def fetch_profile(cls, dtype: Any) -> type[ColumnProfile]:
        stringified: str = str(dtype).lower()
        for _type in cls:
            inds: tuple[str, ...] = _type.value
            is_match: bool = any(ind for ind in inds if ind in stringified)
            if is_match:
                return cls.fetch_prof_map()[_type]
        raise NotImplementedError  # pragma: no-cover


class _ColumnProfileABC(ABC):
    @abstractmethod
    def calc_stats(self, data: DataFrame) -> None: ...

    @abstractmethod
    def _proc_as_html(self) -> dict: ...  # TODO: type hint this


@dataclass
class ColumnProfile(_ColumnProfileABC):
    colname: str
    coltype: str
    sample_data: Any | None = None
    f_unique_vals: float = -1
    f_missing_vals: float = -1

    @property
    def n_missing_vals(self) -> int:
        return self._n_missing_vals

    @n_missing_vals.setter
    def n_missing_vals(self, value: Any) -> None:
        intified: int = int(value)
        self._n_missing_vals = intified
        if intified > 0 and self.n_unique_vals > 0:
            self.n_unique_vals -= 1

    @property
    def n_unique_vals(self) -> int:
        return self._n_unique_vals

    @n_unique_vals.setter
    def n_unique_vals(self, value: Any) -> None:
        self._n_unique_vals = int(value)

    def _fetch_public_attrs(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def spawn_profile(self, _subprofile: type[ColumnProfile]) -> ColumnProfile:
        if type[_subprofile] == type[ColumnProfile]:
            return self  # spawn self if no subprofile
        attrs = self._fetch_public_attrs()
        return _subprofile(**attrs)

    def _calc_general_sats(self, col_data: DataFrame):
        res: dict[str, Series[Any]] = col_data.select(
            _mean=nw.all().mean(),
            _std=nw.all().std(),
            _min=nw.all().min(),
            _max=nw.all().max(),
            _p05=nw.all().quantile(0.005, interpolation="linear"),
            _q_1=nw.all().quantile(0.25, interpolation="linear"),
            _med=nw.all().median(),
            _q_3=nw.all().quantile(0.75, interpolation="linear"),
            _p95=nw.all().quantile(0.95, interpolation="linear"),
        ).to_dict()

        ## Assign stats directly from the data:
        for attr, series in res.items():
            public_name = attr[1:]
            val: float | None = None
            with contextlib.suppress(TypeError, ValueError):
                val: float = float(series.item())
            setattr(self, public_name, val)

        ## Post-hoc calculations:
        ## This is a space to make calculation off of the calculations
        try:
            self.iqr = self.q_3 - self.q_1
        except TypeError:  # if quartiles could not be calculated
            self.iqr = None

    def calc_stats(self, data: DataFrame) -> None:  # pragma: no-cover
        raise NotImplementedError

    def _proc_as_html(self):  # pragma: no-cover
        column_number = column_data["column_number"]
        column_name = column_data["column_name"]
        column_type = column_data["column_type"]

        column_name_and_type = (
            f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
            f"<div style='font-size: 11px; color: gray;'>{column_type}</div>"
        )

        # Get the Missing and Unique value counts and fractions
        missing_vals = column_data["n_missing_values"]
        unique_vals = column_data["n_unique_values"]
        missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
        unique_vals_frac = _compact_decimal_fmt(column_data["f_unique_values"])

        missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"
        unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

        # Create a single dictionary with the statistics for the column
        return {
            "column_number": column_number,
            "icon": SVG_ICONS_FOR_DATA_TYPES["object"],
            "column_name": column_name_and_type,
            "missing_vals": missing_vals_str,
            "unique_vals": unique_vals_str,
            "mean": "&mdash;",
            "std_dev": "&mdash;",
            "min": "&mdash;",
            "p05": "&mdash;",
            "q_1": "&mdash;",
            "med": "&mdash;",
            "q_3": "&mdash;",
            "p95": "&mdash;",
            "max": "&mdash;",
            "iqr": "&mdash;",
        }


class _DateProfile(ColumnProfile):
    min: str | None = None
    max: str | None = None

    def calc_stats(self, data: DataFrame):
        res = data.select(_min=nw.all().min(), _max=nw.all().max()).to_dict()

        ## Pull out elements:
        self.min = str(res["_min"].item())
        self.max = str(res["_max"].item())

    def _proc_as_html(self):
        raise NotImplementedError


class _BoolProfile(ColumnProfile):
    ntrue: int = -1
    nfalse: int = -1

    def calc_stats(self, data: DataFrame):
        self.ntrue: int = int(data.select(nw.all().sum()).item())
        # TODO: document trivalence
        row_count = len(data)
        self.nfalse = row_count - self.ntrue

    def _proc_as_html(self):
        raise NotImplementedError


class _StringProfile(ColumnProfile):
    mean: float = -1
    std: float = -1
    min: float = -1
    max: float = -1
    p05: float = -1
    q_1: float = -1
    med: float = -1
    q_3: float = -1
    p95: float = -1
    iqr: float = -1

    def calc_stats(self, data: DataFrame):
        col_str_len_data = data.select(nw.all().str.len_chars())
        self._calc_general_sats(col_str_len_data)

    def _proc_as_html(self):
        raise NotImplementedError


class _NumericProfile(ColumnProfile):
    n_negative_vals: int = 0
    f_negative_vals: float = 0.0
    n_zero_vals: int = 0
    f_zero_vals: float = 0.0
    n_positive_vals: int = 0
    f_positive_vals: float = 0.0
    mean: float = -1
    std: float = -1
    min: float = -1
    max: float = -1
    p05: float = -1
    q_1: float = -1
    med: float = -1
    q_3: float = -1
    p95: float = -1
    iqr: float = -1

    def calc_stats(self, data: DataFrame):
        self._calc_general_sats(data)

    def _proc_as_html(self):
        raise NotImplementedError


class _DataProfile(NamedTuple):  # TODO: feels redundant
    table_name: str | None
    row_count: int
    columns: list[str] = []
    column_profiles: list[ColumnProfile] = []
