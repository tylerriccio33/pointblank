from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict

import narwhals as nw

from pointblank._constants import SVG_ICONS_FOR_DATA_TYPES
from pointblank._utils import _pivot_to_dict

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.series import Series


## Columns that pose risks for exceptions or don't have handlers
ILLEGAL_TYPES = ("struct",)


class _Metadata(TypedDict):
    row_count: int
    implementation: nw.Implementation


class _TypeMap(Enum):  # ! ordered
    NUMERIC = ("int", "float")
    STRING = ("string", "categorical")
    DATE = ("date",)
    BOOL = ("bool",)

    @classmethod
    def is_illegal(cls, dtype: Any) -> bool:
        return any(ind for ind in ILLEGAL_TYPES if ind in str(dtype).lower())

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

    @classmethod
    def fetch_icon(cls, _type: _TypeMap) -> str:
        icon_map = {
            cls.NUMERIC: "numeric",
            cls.STRING: "string",
            cls.DATE: "date",
            cls.BOOL: "boolean",
        }
        try:
            icon_key = icon_map[_type]
        except KeyError:
            icon_key = "object"
        return SVG_ICONS_FOR_DATA_TYPES[icon_key]


class _ColumnProfileABC(ABC):
    @abstractmethod  # TODO: This should be a class variable
    def fetch_stat_cols(self) -> tuple[str, ...]: ...

    @abstractmethod
    def calc_stats(self, data: DataFrame) -> None: ...


@dataclass
class ColumnProfile(_ColumnProfileABC):
    colname: str
    coltype: str
    sample_data: Any | None = None
    # TODO: Solve the -1 problem generally
    # must exist in the initializer
    _n_missing_vals: int = -1
    _n_unique_vals: int = -1
    # _type: _TypeMap | None = None  # TODO: This should be a class variable

    def fetch_stat_cols(self) -> tuple[str, ...]:
        return ("n_missing_vals", "n_unique_vals")

    def to_df_row(self, metadata: _Metadata, *, format_html: bool = False):
        all_stats = self._fetch_public_attrs()

        # add properties
        # TODO: this won't hold long term
        props = {
            "n_missing_vals": self.n_missing_vals,
            "n_unique_vals": self.n_unique_vals,
        }
        all_stats.update(props)

        # add icon
        icon: str = _TypeMap.fetch_icon(self._type)

        flat_stats = {k: v for k, v in all_stats.items() if k != "sample_data"} | {"icon": icon}

        sample_data: str = ", ".join(all_stats["sample_data"])

        # cast as primitive types:
        primitive_types = (int, float, str, bool)
        valid_stats = {}
        for col, val in flat_stats.items():
            if val is None:
                continue  # None values have variable defaults which confuse the concat
            is_primitive: bool = any(ind for ind in primitive_types if isinstance(val, ind))
            if is_primitive:
                valid_stats[col] = val
                continue

            # we can stringify these only.
            # designed to avoid NaN, nan, inf, etc. which may be added later.
            # these values will create inadvertent mixed types
            if isinstance(val, nw.dtypes.DType):
                valid_stats[col] = str(val)
                continue

            valid_stats[col] = val

        df = nw.from_dict(data=flat_stats, backend=metadata["implementation"]).with_columns(
            sample_data=nw.lit(sample_data),
        )

        if not format_html:
            return df

        raise

    def calc_frac_missing_vals(self, row_count: int) -> float:
        return self.n_missing_vals / row_count

    def calc_frac_unique_vals(self, row_count: int) -> float:
        return self.n_unique_vals / row_count

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
            if callable(v):
                continue
            if isinstance(v, property):
                kprivate: str = "_" + k
                attrs[kprivate] = getattr(self, k)
                continue
            if not k.startswith("_"):
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


class _DateProfile(ColumnProfile):
    min: str | None = None
    max: str | None = None
    _type: _TypeMap = _TypeMap.DATE

    def fetch_stat_cols(self) -> tuple[str, ...]:
        date_stats = ("min", "max")
        return date_stats + super().fetch_stat_cols()

    def calc_stats(self, data: DataFrame):
        res = data.select(_min=nw.all().min(), _max=nw.all().max()).to_dict()

        ## Pull out elements:
        self.min = str(res["_min"].item())
        self.max = str(res["_max"].item())


class _BoolProfile(ColumnProfile):
    ntrue: int = -1
    nfalse: int = -1
    _type: _TypeMap = _TypeMap.BOOL

    def fetch_stat_cols(self) -> tuple[str, ...]:
        bool_stat_cols = ("ntrue", "nfalse")
        return bool_stat_cols + super().fetch_stat_cols()

    def calc_stats(self, data: DataFrame):
        self.ntrue: int = int(data.select(nw.all().sum()).item())
        # TODO: document trivalence
        row_count = len(data)
        self.nfalse = row_count - self.ntrue


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
    _type: _TypeMap = _TypeMap.STRING

    def fetch_stat_cols(self) -> tuple[str, ...]:
        str_stat_cols = ("mean", "std", "min", "max", "p05", "q_1", "med", "q_3", "p95", "iqr")
        return super().fetch_stat_cols() + str_stat_cols

    def calc_stats(self, data: DataFrame):
        # cast as string to avoid ambiguity in cat/enum/etc.
        col_str_len_data = data.select(nw.all().cast(nw.String).str.len_chars())
        self._calc_general_sats(col_str_len_data)


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
    _type: _TypeMap = _TypeMap.NUMERIC

    def fetch_stat_cols(self):
        numeric_stat_cols = (
            "n_negative_vals",
            "f_negative_vals",
            "n_zero_vals",
            "f_zero_vals",
            "n_positive_vals",
            "f_positive_vals",
            "mean",
            "std",
            "min",
            "max",
            "p05",
            "q_1",
            "med",
            "q_3",
            "p95",
            "iqr",
        )
        return super().fetch_stat_cols() + numeric_stat_cols

    def calc_stats(self, data: DataFrame):
        self._calc_general_sats(data)


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

        # TODO: I think this is overly complex

        col_dict = {col.colname: col._fetch_public_attrs() for col in self.column_profiles}

        result_dict = _pivot_to_dict(col_dict)

        return nw.from_dict(result_dict, backend=self.implementation)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<_DataProfile(table_name={self.table_name}, row_count={self.row_count}, columns={self.columns})>"


def _fmt_col_header(name: str, _type: str) -> str:
    return (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{name!s}</div>"
        f"<div style='font-size: 11px; color: gray;'>{_type!s}</div>"
    )
