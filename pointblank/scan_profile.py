from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import narwhals as nw

from pointblank._constants import SVG_ICONS_FOR_DATA_TYPES
from pointblank._datascan_utils import _compact_decimal_fmt, _compact_integer_fmt

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

    def _fetch_public_attrs(self) -> dict:  # TODO: generic type hint
        # Get all attributes, including properties
        attrs = {}
        for k, v in vars(self.__class__).items():
            if isinstance(v, property) and not k.startswith("_"):
                attrs[k] = getattr(self, k)
        # Add non-private instance attributes
        attrs.update({k: v for k, v in vars(self).items() if not k.startswith("_")})
        return attrs

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

    def _proc_as_html(self):
        missing_vals_str = f"{self.n_missing_vals}<br>{self.f_missing_vals}"
        unique_vals_str = f"{self.n_unique_vals}<br>{self.f_unique_vals}"

        # Create a single dictionary with the statistics for the column
        # TODO: This should be declaritively typed
        return {
            "icon": SVG_ICONS_FOR_DATA_TYPES["object"],
            "column_name": _fmt_col_header(self.colname, self.coltype),
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
        column_number = column_data["column_number"]

        column_name = column_data["column_name"]
        column_type = column_data["column_type"]

        long_column_type = len(column_type) > 22

        if long_column_type:
            column_type_style = (
                "font-size: 7.5px; color: gray; margin-top: 3px; margin-bottom: 2px;"
            )
        else:
            column_type_style = "font-size: 11px; color: gray;"

        column_name_and_type = (
            f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
            f"<div style='{column_type_style}'>{column_type}</div>"
        )

        # Get the Missing and Unique value counts and fractions
        missing_vals = column_data["n_missing_values"]
        unique_vals = column_data["n_unique_values"]
        missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
        unique_vals_frac = _compact_decimal_fmt(column_data["f_unique_values"])

        missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"
        unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

        # Get the min and max date
        min_date = column_data["statistics"]["datetime"]["min"]
        max_date = column_data["statistics"]["datetime"]["max"]

        # Format the dates so that they don't break across lines
        min_max_date_str = f"<span style='text-align: left; white-space: nowrap; overflow-x: visible;'>&nbsp;{min_date} &ndash; {max_date}</span>"

        # Create a single dictionary with the statistics for the column
        return {
            "column_number": column_number,
            "icon": SVG_ICONS_FOR_DATA_TYPES["date"],
            "column_name": column_name_and_type,
            "missing_vals": missing_vals_str,
            "unique_vals": unique_vals_str,
            "mean": "&mdash;",
            "std_dev": "&mdash;",
            "min": min_max_date_str,
            "p05": "",
            "q_1": "",
            "med": "",
            "q_3": "",
            "p95": "",
            "max": "",
            "iqr": "&mdash;",
        }


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
        # cast as string to avoid ambiguity in cat/enum/etc.
        col_str_len_data = data.select(nw.all().cast(nw.String).str.len_chars())
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
        icon = "numeric"  # TODO: should this be here

        missing_vals_frac = _compact_decimal_fmt(self.f_missing_vals)
        unique_vals_frac = _compact_decimal_fmt(self.f_unique_vals)

        missing_vals_str = f"{self.n_missing_vals}<br>{missing_vals_frac}"
        unique_vals_str = f"{self.n_unique_vals}<br>{unique_vals_frac}"

        # Get the descriptive and quantile statistics
        descriptive_stats = column_data["statistics"][key]["descriptive"]
        quantile_stats = column_data["statistics"][key]["quantiles"]

        # Get all values from the descriptive and quantile stats into a single list
        quantile_stats_vals = [v[1] for v in quantile_stats.items()]

        # Determine if the quantile stats are all integerlike
        integerlike = []

        # Determine if the quantile stats are integerlike
        for val in quantile_stats_vals:
            # Check if a quantile value is a number and then if it is intergerlike
            if not isinstance(val, (int, float)):
                continue
            else:
                integerlike.append(val % 1 == 0)
        quantile_vals_integerlike = all(integerlike)

        # Determine the formatter to use for the quantile values
        if quantile_vals_integerlike:
            q_formatter = _compact_integer_fmt
        else:
            q_formatter = _compact_decimal_fmt

        # Format the descriptive statistics (mean and standard deviation)
        for key, value in descriptive_stats.items():
            descriptive_stats[key] = _compact_decimal_fmt(value=value)

        # Format the quantile statistics
        for key, value in quantile_stats.items():
            quantile_stats[key] = q_formatter(value=value)

        # Create a single dictionary with the statistics for the column
        return {
            "icon": SVG_ICONS_FOR_DATA_TYPES[icon],
            "column_name": column_name_and_type,
            "missing_vals": missing_vals_str,
            "unique_vals": unique_vals_str,
            **descriptive_stats,
            **quantile_stats,
        }


class _DataProfile:  # TODO: feels redundant and weird
    def __init__(
        self,
        table_name: str | None,
        row_count: int,
        columns: list[str],
        implementation: nw.Implementation,
    ):
        self.table_name: str | None = table_name
        self.row_count: int = row_count
        self.columns: list[str] = columns
        self.implementation = implementation
        self.column_profiles: list[ColumnProfile] = []

    def as_dataframe(self) -> DataFrame:
        ## Convert list[dict] to dict[list]:
        ## We remove nested types like `sample_data`.
        ## We stringify all non-primitive types.
        from collections import defaultdict

        col_dict = {col.colname: col._fetch_public_attrs() for col in self.column_profiles}
        nested_types = ("sample_data",)
        prim_types = (int, str, float)
        for col in col_dict:
            for nested_type in nested_types:
                col_dict[col].pop(nested_type, None)
            for key in list(col_dict[col].keys()):
                if not isinstance(col_dict[col][key], prim_types):
                    col_dict[col][key] = str(col_dict[col][key])

        dict_of_lists = defaultdict(list)
        for col_data in col_dict.values():
            for key, value in col_data.items():
                dict_of_lists[key].append(value)
        dict_of_lists = dict(dict_of_lists)

        return nw.from_dict(dict_of_lists, backend=self.implementation)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<_DataProfile(table_name={self.table_name}, row_count={self.row_count}, columns={self.columns})>"


def _fmt_col_header(name: str, _type: str) -> str:
    return (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{name!s}</div>"
        f"<div style='font-size: 11px; color: gray;'>{_type!s}</div>"
    )
