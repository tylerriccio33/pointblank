from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.series import Series


class _TypeMap(Enum):
    NUMERIC = ("int", "float")
    STRING = ("string", "categorical")
    DATE = ("date",)
    BOOL = ("bool",)

    @classmethod
    def fetch_prof_map(cls) -> dict[_TypeMap, type[ColumnProfile]]:
        return {
            cls.BOOL: _BoolProfile,
            cls.NUMERIC: _NumericProfile,
            cls.STRING: _StringProfile,
            cls.DATE: _DateProfile,
        }

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


@dataclass
class ColumnProfile(_ColumnProfileABC):
    colname: str
    colnumber: int
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
            val: float = float(series.item())
            setattr(self, public_name, val)

        ## Post-hoc calculations:
        ## This is a space to make calculation off of the calculations
        self.iqr = self.q_3 - self.q_1

    def calc_stats(self, data: DataFrame) -> None:  # pragma: no-cover
        msg = "Statistics should be calculated from subclasses of `ColumnProfile`."
        raise NotImplementedError(msg)


class _DateProfile(ColumnProfile):
    min: str | None = None
    max: str | None = None

    def calc_stats(self, data: DataFrame):
        res = data.select(_min=nw.all().min(), _max=nw.all().max()).to_dict()

        ## Pull out elements:
        self.min = str(res["_min"].item())
        self.max = str(res["_max"].item())


class _BoolProfile(ColumnProfile):
    ntrue: int = -1
    nfalse: int = -1

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

    def calc_stats(self, data: DataFrame):
        col_str_len_data = data.select(nw.all().str.len_chars())
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

    def calc_stats(self, data: DataFrame):
        self._calc_general_sats(data)


class _DataProfile(NamedTuple):  # TODO: feels redundant
    table_name: str | None
    row_count: int
    columns: list[str] = []
    column_profiles: list[ColumnProfile] = []
