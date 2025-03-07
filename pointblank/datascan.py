from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import floor, log10
from typing import Any

import narwhals as nw
from narwhals.typing import FrameT

from pointblank._utils import _get_tbl_type, _select_df_lib

__all__ = [
    "DataScan",
]


@dataclass
class DataScan:
    """
    Get a summary of a dataset.

    The `DataScan` class provides a way to get a summary of a dataset. The summary includes the
    following information:

    - the name of the table (if provided)
    - the type of the table (e.g., `"polars"`, `"pandas"`, etc.)
    - the number of rows and columns in the table
    - column-level information, including:
        - the column name
        - the column type
        - measures of missingness and distinctness
        - measures of negative, zero, and positive values (for numerical columns)
        - a sample of the data (the first 5 values)
        - statistics (if the column contains numbers, strings, or datetimes)

    To obtain a dictionary representation of the summary, you can use the `to_dict()` method. To
    get a JSON representation of the summary, you can use the `to_json()` method. To save the JSON
    text to a file, the `save_to_json()` method could be used.

    :::{.callout-warning}
    The `DataScan()` class is still experimental. Please report any issues you encounter in the
    [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).
    :::

    Parameters
    ----------
    data
        The data to scan and summarize.
    tbl_name
        Optionally, the name of the table could be provided as `tbl_name`.

    Measures of Missingness and Distinctness
    ----------------------------------------
    For each column, the following measures are provided:

    - `n_missing_values`: the number of missing values in the column
    - `f_missing_values`: the fraction of missing values in the column
    - `n_unique_values`: the number of unique values in the column
    - `f_unique_values`: the fraction of unique values in the column

    The fractions are calculated as the ratio of the measure to the total number of rows in the
    dataset.

    Counts and Fractions of Negative, Zero, and Positive Values
    -----------------------------------------------------------
    For numerical columns, the following measures are provided:

    - `n_negative_values`: the number of negative values in the column
    - `f_negative_values`: the fraction of negative values in the column
    - `n_zero_values`: the number of zero values in the column
    - `f_zero_values`: the fraction of zero values in the column
    - `n_positive_values`: the number of positive values in the column
    - `f_positive_values`: the fraction of positive values in the column

    The fractions are calculated as the ratio of the measure to the total number of rows in the
    dataset.

    Statistics for Numerical Columns
    --------------------------------
    For numerical columns, the following descriptive statistics are provided:

    - `mean`: the mean of the column
    - `std_dev`: the standard deviation of the column

    Additionally, the following quantiles are provided:

    - `min`: the minimum value in the column
    - `p05`: the 5th percentile of the column
    - `q_1`: the first quartile of the column
    - `med`: the median of the column
    - `q_3`: the third quartile of the column
    - `p95`: the 95th percentile of the column
    - `max`: the maximum value in the column
    - `iqr`: the interquartile range of the column

    Statistics for String Columns
    -----------------------------
    For string columns, the following statistics are provided:

    - `mode`: the mode of the column

    Statistics for Datetime Columns
    -------------------------------
    For datetime columns, the following statistics are provided:

    - `min_date`: the minimum date in the column
    - `max_date`: the maximum date in the column

    Returns
    -------
    DataScan
        A DataScan object.
    """

    data: FrameT | Any
    tbl_name: str | None = None
    data_alt: Any | None = field(init=False)
    tbl_category: str = field(init=False)
    tbl_type: str = field(init=False)
    profile: dict = field(init=False)

    def __post_init__(self):
        # Determine if the data is a DataFrame that could be handled by Narwhals,
        # or an Ibis Table
        self.tbl_type = _get_tbl_type(data=self.data)
        ibis_tbl = "ibis.expr.types.relations.Table" in str(type(self.data))
        pl_pd_tbl = "polars" in self.tbl_type or "pandas" in self.tbl_type

        # Set the table category based on the type of table (this will be used to determine
        # how to handle the data)
        if ibis_tbl:
            self.tbl_category = "ibis"
        else:
            self.tbl_category = "dataframe"

        # If the data is DataFrame, convert it to a Narwhals DataFrame
        if pl_pd_tbl:
            self.data_alt = nw.from_native(self.data)
        else:
            self.data_alt = None

        # Generate the profile based on the `tbl_category` value
        if self.tbl_category == "dataframe":
            self.profile = self._generate_profile_df()

        if self.tbl_category == "ibis":
            self.profile = self._generate_profile_ibis()

    def _generate_profile_df(self) -> dict:
        profile = {}

        if self.tbl_name:
            profile["tbl_name"] = self.tbl_name

        row_count = self.data_alt.shape[0]
        column_count = self.data_alt.shape[1]

        profile.update(
            {
                "tbl_type": self.tbl_type,
                "dimensions": {"rows": row_count, "columns": column_count},
                "columns": [],
            }
        )

        for column in self.data_alt.columns:
            col_data = self.data_alt[column]
            native_dtype = str(self.data[column].dtype)

            if "date" in str(col_data.dtype).lower():
                sample_data = col_data.drop_nulls().head(5).cast(nw.String).to_list()
                sample_data = [str(x) for x in sample_data]
            else:
                sample_data = col_data.drop_nulls().head(5).to_list()

            n_missing_vals = int(col_data.is_null().sum())
            n_unique_vals = int(col_data.n_unique())

            # If there are missing values, subtract 1 from the number of unique values
            # to account for the missing value which shouldn't be included in the count
            if (n_missing_vals > 0) and (n_unique_vals > 0):
                n_unique_vals = n_unique_vals - 1

            f_missing_vals = _round_to_sig_figs(n_missing_vals / row_count, 3)
            f_unique_vals = _round_to_sig_figs(n_unique_vals / row_count, 3)

            col_profile = {
                "column_name": column,
                "column_type": native_dtype,
                "n_missing_values": n_missing_vals,
                "f_missing_values": f_missing_vals,
                "n_unique_values": n_unique_vals,
                "f_unique_values": f_unique_vals,
            }

            if "int" in str(col_data.dtype).lower() or "float" in str(col_data.dtype).lower():
                n_negative_vals = int(col_data.is_between(-1e26, -1e-26).sum())
                f_negative_vals = _round_to_sig_figs(n_negative_vals / row_count, 3)

                n_zero_vals = int(col_data.is_between(0, 0).sum())
                f_zero_vals = _round_to_sig_figs(n_zero_vals / row_count, 3)

                n_positive_vals = row_count - n_missing_vals - n_negative_vals - n_zero_vals
                f_positive_vals = _round_to_sig_figs(n_positive_vals / row_count, 3)

                col_profile_additional = {
                    "n_negative_values": n_negative_vals,
                    "f_negative_values": f_negative_vals,
                    "n_zero_values": n_zero_vals,
                    "f_zero_values": f_zero_vals,
                    "n_positive_values": n_positive_vals,
                    "f_positive_values": f_positive_vals,
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                col_profile_stats = {
                    "statistics": {
                        "numerical": {
                            "descriptive": {
                                "mean": round(float(col_data.mean()), 2),
                                "std_dev": round(float(col_data.std()), 4),
                            },
                            "quantiles": {
                                "min": float(col_data.min()),
                                "p05": round(
                                    float(col_data.quantile(0.05, interpolation="linear")), 2
                                ),
                                "q_1": round(
                                    float(col_data.quantile(0.25, interpolation="linear")), 2
                                ),
                                "med": float(col_data.median()),
                                "q_3": round(
                                    float(col_data.quantile(0.75, interpolation="linear")), 2
                                ),
                                "p95": round(
                                    float(col_data.quantile(0.95, interpolation="linear")), 2
                                ),
                                "max": float(col_data.max()),
                                "iqr": round(
                                    float(col_data.quantile(0.75, interpolation="linear"))
                                    - float(col_data.quantile(0.25, interpolation="linear")),
                                    2,
                                ),
                            },
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            elif (
                "string" in str(col_data.dtype).lower()
                or "categorical" in str(col_data.dtype).lower()
            ):
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                mode = col_data.mode()

                col_profile_stats = {
                    "statistics": {
                        "string": {
                            "mode": mode[0],
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            elif "date" in str(col_data.dtype).lower():
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                min_date = str(col_data.min())
                max_date = str(col_data.max())

                col_profile_stats = {
                    "statistics": {
                        "datetime": {
                            "min": min_date,
                            "max": max_date,
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            profile["columns"].append(col_profile)

        return profile

    def _generate_profile_ibis(self) -> dict:
        profile = {}

        if self.tbl_name:
            profile["tbl_name"] = self.tbl_name

        from pointblank.validate import get_row_count

        row_count = get_row_count(data=self.data)
        column_count = len(self.data.columns)

        profile.update(
            {
                "tbl_type": self.tbl_type,
                "dimensions": {"rows": row_count, "columns": column_count},
                "columns": [],
            }
        )

        # Determine which DataFrame library is available
        df_lib = _select_df_lib(preference="polars")
        df_lib_str = str(df_lib)

        if "polars" in df_lib_str:
            df_lib_use = "polars"
        else:
            df_lib_use = "pandas"

        column_dtypes = list(self.data.schema().items())

        for idx, column in enumerate(self.data.columns):
            dtype_str = str(column_dtypes[idx][1])

            col_data = self.data[column]
            col_data_no_null = self.data.drop_null().head(5)[column]

            if "date" in dtype_str.lower() or "timestamp" in dtype_str.lower():
                if df_lib_use == "polars":
                    import polars as pl

                    sample_data = col_data_no_null.to_polars().cast(pl.String).to_list()
                else:
                    sample_data = col_data_no_null.to_pandas().astype(str).to_list()
            else:
                if df_lib_use == "polars":
                    sample_data = col_data_no_null.to_polars().to_list()
                else:
                    sample_data = col_data_no_null.to_pandas().to_list()

            n_missing_vals = int(_to_df_lib(col_data.isnull().sum(), df_lib=df_lib_use))
            n_unique_vals = int(_to_df_lib(col_data.nunique(), df_lib=df_lib_use))

            # If there are missing values, subtract 1 from the number of unique values
            # to account for the missing value which shouldn't be included in the count
            if (n_missing_vals > 0) and (n_unique_vals > 0):
                n_unique_vals = n_unique_vals - 1

            f_missing_vals = _round_to_sig_figs(n_missing_vals / row_count, 3)
            f_unique_vals = _round_to_sig_figs(n_unique_vals / row_count, 3)

            col_profile = {
                "column_name": column,
                "column_type": dtype_str,
                "n_missing_values": n_missing_vals,
                "f_missing_values": f_missing_vals,
                "n_unique_values": n_unique_vals,
                "f_unique_values": f_unique_vals,
            }

            if "int" in dtype_str.lower() or "float" in dtype_str.lower():
                n_negative_vals = int(
                    _to_df_lib(col_data.between(-1e26, -1e-26).sum(), df_lib=df_lib_use)
                )
                f_negative_vals = _round_to_sig_figs(n_negative_vals / row_count, 3)

                n_zero_vals = int(_to_df_lib(col_data.between(0, 0).sum(), df_lib=df_lib_use))
                f_zero_vals = _round_to_sig_figs(n_zero_vals / row_count, 3)

                n_positive_vals = row_count - n_missing_vals - n_negative_vals - n_zero_vals
                f_positive_vals = _round_to_sig_figs(n_positive_vals / row_count, 3)

                col_profile_additional = {
                    "n_negative_values": n_negative_vals,
                    "f_negative_values": f_negative_vals,
                    "n_zero_values": n_zero_vals,
                    "f_zero_values": f_zero_vals,
                    "n_positive_values": n_positive_vals,
                    "f_positive_values": f_positive_vals,
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                col_profile_stats = {
                    "statistics": {
                        "numerical": {
                            "descriptive": {
                                "mean": round(_to_df_lib(col_data.mean(), df_lib=df_lib_use), 2),
                                "std_dev": round(_to_df_lib(col_data.std(), df_lib=df_lib_use), 4),
                            },
                            "quantiles": {
                                "min": _to_df_lib(col_data.min(), df_lib=df_lib_use),
                                "p05": round(
                                    _to_df_lib(col_data.approx_quantile(0.05), df_lib=df_lib_use),
                                    2,
                                ),
                                "q_1": round(
                                    _to_df_lib(col_data.approx_quantile(0.25), df_lib=df_lib_use),
                                    2,
                                ),
                                "med": _to_df_lib(col_data.median(), df_lib=df_lib_use),
                                "q_3": round(
                                    _to_df_lib(col_data.approx_quantile(0.75), df_lib=df_lib_use),
                                    2,
                                ),
                                "p95": round(
                                    _to_df_lib(col_data.approx_quantile(0.95), df_lib=df_lib_use),
                                    2,
                                ),
                                "max": _to_df_lib(col_data.max(), df_lib=df_lib_use),
                                "iqr": round(
                                    _to_df_lib(col_data.quantile(0.75), df_lib=df_lib_use)
                                    - _to_df_lib(col_data.quantile(0.25), df_lib=df_lib_use),
                                    2,
                                ),
                            },
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            elif "string" in dtype_str.lower() or "char" in dtype_str.lower():
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                col_profile_stats = {
                    "statistics": {
                        "string": {
                            "mode": _to_df_lib(col_data.mode(), df_lib=df_lib_use),
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            elif "date" in dtype_str.lower() or "timestamp" in dtype_str.lower():
                col_profile_additional = {
                    "sample_data": sample_data,
                }
                col_profile.update(col_profile_additional)

                min_date = _to_df_lib(col_data.min(), df_lib=df_lib_use)
                max_date = _to_df_lib(col_data.max(), df_lib=df_lib_use)

                col_profile_stats = {
                    "statistics": {
                        "datetime": {
                            "min_date": str(min_date),
                            "max_date": str(max_date),
                        }
                    }
                }
                col_profile.update(col_profile_stats)

            profile["columns"].append(col_profile)

        return profile

    def _get_column_data(self, column: str) -> dict | None:
        column_data = self.profile["columns"]

        # Find the column in the column data and return the
        for col in column_data:
            if col["column_name"] == column:
                return col

        # If the column is not found, return None
        return None

    def to_dict(self) -> dict:
        return self.profile

    def to_json(self) -> str:
        return json.dumps(self.profile, indent=4)

    def save_to_json(self, output_file: str):
        with open(output_file, "w") as f:
            json.dump(self.profile, f, indent=4)


def _to_df_lib(expr: any, df_lib: str) -> any:
    if df_lib == "polars":
        return expr.to_polars()
    else:
        return expr.to_pandas()


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    if value == 0:
        return 0
    return round(value, sig_figs - int(floor(log10(abs(value)))) - 1)
