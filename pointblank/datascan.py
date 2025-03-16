from __future__ import annotations

import contextlib
import json
import warnings
from importlib.metadata import version
from math import floor, log10
from typing import TYPE_CHECKING, Any

import narwhals as nw
from great_tables import GT, google_font, html, loc, style
from great_tables.vals import fmt_integer, fmt_number, fmt_scientific
from narwhals.typing import FrameT

from pointblank._constants import SVG_ICONS_FOR_DATA_TYPES
from pointblank._utils import _select_df_lib
from pointblank._utils_html import _create_table_dims_html, _create_table_type_html
from pointblank.scan_profile import ColumnProfile, _DataProfile, _TypeMap

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.dataframe import DataFrame
    from narwhals.typing import CompliantDataFrame, IntoDataFrame


__all__ = ["DataScan", "col_summary_tbl"]


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

    Statistics for Numerical and String Columns
    -------------------------------------------
    For numerical and string columns, several statistical measures are provided. Please note that
    for string columms, the statistics are based on the lengths of the strings in the column.

    The following descriptive statistics are provided:

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

    Statistics for Date and Datetime Columns
    ----------------------------------------
    For date/datetime columns, the following statistics are provided:

    - `min`: the minimum date/datetime in the column
    - `max`: the maximum date/datetime in the column

    Returns
    -------
    DataScan
        A DataScan object.
    """

    def __init__(
        self, data: IntoDataFrame, tbl_name: str | None = None, *, force_collect: bool = True
    ) -> None:
        self.nw_data: DataFrame = nw.from_native(data)

        # # TODO: this must be wrong, investigate a more idiomatic way
        is_lazy = self.nw_data._level == "lazy"
        if is_lazy and not force_collect:  # ? this should be allowed (eventually?)
            msg = (
                "`DataScan` requires a dataframe to avoid unexpected computations "
                "of the caller's execution graph. Please collect the data into a "
                "narwhals compliant dataframe first or turn on `force_collect`."
            )
            raise TypeError(msg)
        if is_lazy and force_collect:
            self.nw_data = self.nw_data.collect()

        self.tbl_name: str | None = tbl_name
        self.profile: _DataProfile = self._generate_profile_df()

    def _generate_profile_df(self) -> _DataProfile:
        row_count = len(self.nw_data)

        columns: list[str] = self.nw_data.columns

        profile = _DataProfile(table_name=self.tbl_name, row_count=row_count, columns=columns)

        schema: Mapping[str, Any] = self.nw_data.schema
        for column in columns:
            col_data: DataFrame = self.nw_data.select(column)

            ## Handle dtyping:
            native_dtype = schema[column]
            try:
                prof: type[ColumnProfile] = _TypeMap.fetch_profile(native_dtype)
            except NotImplementedError:
                continue

            col_profile = ColumnProfile(colname=column, coltype=native_dtype)

            ## Collect Sample Data:
            ## This is the most consistent way (i think) to get the samples out of the data.
            ## We can avoid writing our own logic to determine operations and rely on narwhals.
            raw_vals: list[Any] = col_data.drop_nulls().head(5).to_dict()[column].to_list()
            col_profile.sample_data = [str(x) for x in raw_vals]

            try:
                col_profile.n_unique_vals = col_data.is_unique().sum()  # set this before missing
            except Exception as e:  # tendancy to introduce internal panics
                msg = f"Could not calculate uniqueness and missing values: {e!s}"
                warnings.warn(msg)
            else:
                col_profile.n_missing_vals = col_data.null_count().item()
                # TODO: These should probably live on the class
                col_profile.f_missing_vals = _round_to_sig_figs(
                    col_profile.n_missing_vals / row_count, 3
                )
                col_profile.f_unique_vals = _round_to_sig_figs(
                    col_profile.n_unique_vals / row_count, 3
                )

            sub_profile: ColumnProfile = col_profile.spawn_profile(prof)
            with contextlib.suppress(NotImplementedError):
                sub_profile.calc_stats(col_data)

            profile.column_profiles.append(sub_profile)

        return profile

    def _get_column_data(self, column: str) -> dict | None:
        column_data = self.profile["columns"]

        # Find the column in the column data and return the
        for col in column_data:
            if col["column_name"] == column:
                return col

        # If the column is not found, return None
        return None

    @property
    def summary_data(self) -> CompliantDataFrame:
        try:
            return self._summary_data
        except AttributeError:
            self._gen_stats_list()
            self._summary_data_from_list()
            return self._summary_data

    def _gen_stats_list(self) -> None:
        """Set the `stats_list` attribute."""
        column_profiles = self.profile.column_profiles

        stats_list = []
        datetime_row_list = []

        # Iterate over each column's data and obtain a dictionary of statistics for each column
        for idx, col in enumerate(column_profiles):
            col_dict = col._proc_as_html()

            # if "statistics" in col and (
            #     "numerical" in col["statistics"] or "string_lengths" in col["statistics"]
            # ):
            #     col_dict = _process_numerical_string_column_data(col)
            # elif "statistics" in col and "datetime" in col["statistics"]:
            #     col_dict = _process_datetime_column_data(col)
            #     datetime_row_list.append(idx)
            # elif "statistics" in col and "boolean" in col["statistics"]:
            #     col_dict = _process_boolean_column_data(col)
            # else:
            #     col_dict = _process_other_column_data(col)

            stats_list.append(col_dict)

        self._stats_list = stats_list

    def _summary_data_from_list(self) -> None:
        """Set the `_summary_data` attribute from the raw stats list."""
        raise NotImplementedError

        # Determine which DataFrame library is available and construct the DataFrame
        # based on the available library
        df_lib = _select_df_lib(preference="polars")
        df_lib_str = str(df_lib)

        # TODO: this can probably be simplified
        if "polars" in df_lib_str:
            import polars as pl

            stats_df: CompliantDataFrame = pl.DataFrame(self._stats_list)
        else:
            import pandas as pd

            stats_df: CompliantDataFrame = pd.DataFrame(self._stats_list)

        self._summary_data: CompliantDataFrame = pl.DataFrame(stats_df)

    def get_tabular_report(self) -> GT:
        summary_data = self.summary_data

        n_rows = self.profile["dimensions"]["rows"]
        n_columns = self.profile["dimensions"]["columns"]

        stat_columns = [
            "missing_vals",
            "unique_vals",
            "mean",
            "std_dev",
            "min",
            "p05",
            "q_1",
            "med",
            "q_3",
            "p95",
            "max",
            "iqr",
        ]

        # Create the label, table type, and thresholds HTML fragments
        table_type_html = _create_table_type_html(
            tbl_type=self.tbl_type, tbl_name=self.tbl_name, font_size="10px"
        )

        tbl_dims_html = _create_table_dims_html(columns=n_columns, rows=n_rows, font_size="10px")

        # Compose the subtitle HTML fragment
        combined_title = (
            "<div>"
            '<div style="padding-top: 0; padding-bottom: 7px;">'
            f"{table_type_html}"
            f"{tbl_dims_html}"
            "</div>"
            "</div>"
        )

        # TODO: Ensure width is 905px in total

        gt_tbl = (
            GT(self._summary_data)
            .tab_header(title=html(combined_title))
            .cols_align(align="right", columns=stat_columns)
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(
                style=style.text(font=google_font("IBM Plex Mono")),
                locations=loc.body(),
            )
            .tab_style(
                style=style.text(size="10px"),
                locations=loc.body(columns=stat_columns),
            )
            .tab_style(
                style=style.text(size="14px"),
                locations=loc.body(columns="column_number"),
            )
            .tab_style(
                style=style.text(size="12px"),
                locations=loc.body(columns="column_name"),
            )
            .tab_style(
                style=style.css("white-space: pre; overflow-x: visible;"),
                locations=loc.body(columns="min"),
            )
            .tab_style(
                style=style.borders(sides="left", color="#D3D3D3", style="solid"),
                locations=loc.body(columns=["missing_vals", "mean", "iqr"]),
            )
            .tab_style(
                style=style.borders(sides="left", color="#E5E5E5", style="dashed"),
                locations=loc.body(
                    columns=["std_dev", "min", "p05", "q_1", "med", "q_3", "p95", "max"]
                ),
            )
            .tab_style(
                style=style.borders(sides="left", style="none"),
                locations=loc.body(
                    columns=["p05", "q_1", "med", "q_3", "p95", "max"],
                    rows=self._stats_list,
                ),
            )
            .tab_style(
                style=style.fill(color="#FCFCFC"),
                locations=loc.body(columns=["missing_vals", "unique_vals", "iqr"]),
            )
            .cols_label(
                column_number="",
                icon="",
                column_name="Column",
                missing_vals="NAs",
                unique_vals="Uniq.",
                mean="Mean",
                std_dev="S.D.",
                min="Min",
                p05="P05",
                q_1="Q1",
                med="Med",
                q_3="Q3",
                p95="P95",
                max="Max",
                iqr="IQR",
            )
            .cols_width(
                column_number="40px",
                icon="35px",
                column_name="200px",
                missing_vals="50px",
                unique_vals="50px",
                mean="50px",
                std_dev="50px",
                min="50px",
                p05="50px",
                q_1="50px",
                med="50px",
                q_3="50px",
                p95="50px",
                max="50px",
                iqr="50px",  # 875 px total
            )
        )

        # If the version of `great_tables` is `>=0.17.0` then disable Quarto table processing
        if version("great_tables") >= "0.17.0":
            gt_tbl = gt_tbl.tab_options(quarto_disable_processing=True)

        return gt_tbl

    def to_dict(self) -> dict:
        return self.profile

    def to_json(self) -> str:
        profiles: list[dict] = []
        for profile in self.profile.column_profiles:
            attrs = profile._fetch_public_attrs()
            profiles.append(attrs)

        return json.dumps(profiles, indent=4, default=str)

    def save_to_json(self, output_file: str):
        json_string: str = self.to_json()
        with open(output_file, "w") as f:
            json.dump(json_string, f, indent=4)


def col_summary_tbl(data: FrameT | Any, tbl_name: str | None = None) -> GT:
    """
    Generate a column-level summary table of a dataset.

    The `col_summary_tbl()` function generates a summary table of a dataset, focusing on providing
    column-level information about the dataset. The summary includes the following information:

    - the type of the table (e.g., `"polars"`, `"pandas"`, etc.)
    - the number of rows and columns in the table
    - column-level information, including:
        - the column name
        - the column type
        - measures of missingness and distinctness
        - descriptive stats and quantiles
        - statistics for datetime columns

    The summary table is returned as a GT object, which can be displayed in a notebook or saved to
    an HTML file.

    :::{.callout-warning}
    The `col_summary_tbl()` function is still experimental. Please report any issues you encounter
    in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).
    :::

    Parameters
    ----------
    data
        The table to summarize, which could be a DataFrame object or an Ibis table object. Read the
        *Supported Input Table Types* section for details on the supported table types.
    tbl_name
        Optionally, the name of the table could be provided as `tbl_name=`.

    Returns
    -------
    GT
        A GT object that displays the column-level summaries of the table.

    Supported Input Table Types
    ---------------------------
    The `data=` parameter can be given any of the following table types:

    - Polars DataFrame (`"polars"`)
    - Pandas DataFrame (`"pandas"`)
    - DuckDB table (`"duckdb"`)*
    - MySQL table (`"mysql"`)*
    - PostgreSQL table (`"postgresql"`)*
    - SQLite table (`"sqlite"`)*
    - Parquet table (`"parquet"`)*

    The table types marked with an asterisk need to be prepared as Ibis tables (with type of
    `ibis.expr.types.relations.Table`). Furthermore, using `col_summary_tbl()` with these types of
    tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a
    Polars or Pandas DataFrame, the availability of Ibis is not needed.

    Examples
    --------
    It's easy to get a column-level summary of a table using the `col_summary_tbl()` function.
    Here's an example using the `small_table` dataset (itself loaded using the
    [`load_dataset()`](`pointblank.load_dataset`) function):

    ```{python}
    import pointblank as pb

    small_table_polars = pb.load_dataset(dataset="small_table", tbl_type="polars")

    pb.col_summary_tbl(data=small_table_polars)
    ```

    This table used above was a Polars DataFrame, but the `col_summary_tbl()` function works with
    any table supported by `pointblank`, including Pandas DataFrames and Ibis backend tables.
    Here's an example using a DuckDB table handled by Ibis:

    ```{python}
    small_table_duckdb = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

    pb.col_summary_tbl(data=small_table_duckdb, tbl_name="nycflights")
    ```
    """

    scanner = DataScan(data=data, tbl_name=tbl_name)
    return scanner.get_tabular_report()


def _to_df_lib(expr: any, df_lib: str) -> any:
    if df_lib == "polars":
        return expr.to_polars()
    else:
        return expr.to_pandas()


def _round_to_sig_figs(value: float, sig_figs: int) -> float:
    if value == 0:
        return 0
    return round(value, sig_figs - int(floor(log10(abs(value)))) - 1)


def _compact_integer_fmt(value: float | int) -> str:
    if value == 0:
        formatted = "0"
    elif abs(value) >= 1 and abs(value) < 10_000:
        formatted = fmt_integer(value, use_seps=False)[0]
    else:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]

    return formatted


def _compact_decimal_fmt(value: float | int) -> str:
    if value == 0:
        formatted = "0.00"
    elif abs(value) < 1 and abs(value) >= 0.01:
        formatted = fmt_number(value, decimals=2)[0]
    elif abs(value) < 0.01:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]
    elif abs(value) >= 1 and abs(value) < 1000:
        formatted = fmt_number(value, n_sigfig=3)[0]
    elif abs(value) >= 1000 and abs(value) < 10_000:
        formatted = fmt_number(value, decimals=0, use_seps=False)[0]
    else:
        formatted = fmt_scientific(value, decimals=1, exp_style="E1")[0]

    return formatted


def _compact_0_1_fmt(value: float | int) -> str:
    if value == 0:
        formatted = " 0.00"
    elif value == 1:
        formatted = " 1.00"
    elif abs(value) < 1 and abs(value) >= 0.01:
        formatted = " " + fmt_number(value, decimals=2)[0]
    elif abs(value) < 0.01:
        formatted = "<0.01"
    elif abs(value) > 0.99:
        formatted = ">0.99"
    else:
        formatted = fmt_number(value, n_sigfig=3)[0]

    return formatted


def _process_numerical_string_column_data(column_data: dict) -> dict:
    column_number = column_data["column_number"]
    column_name = column_data["column_name"]
    column_type = column_data["column_type"]

    column_name_and_type = (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
        f"<div style='font-size: 11px; color: gray;'>{column_type}</div>"
    )

    # Determine if the column is a numerical or string column
    if "numerical" in column_data["statistics"]:
        key = "numerical"
        icon = "numeric"
    elif "string_lengths" in column_data["statistics"]:
        key = "string_lengths"
        icon = "string"

    # Get the Missing and Unique value counts and fractions
    missing_vals = column_data["n_missing_values"]
    unique_vals = column_data["n_unique_values"]
    missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
    unique_vals_frac = _compact_decimal_fmt(column_data["f_unique_values"])

    missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"
    unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

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
    stats_dict = {
        "column_number": column_number,
        "icon": SVG_ICONS_FOR_DATA_TYPES[icon],
        "column_name": column_name_and_type,
        "missing_vals": missing_vals_str,
        "unique_vals": unique_vals_str,
        **descriptive_stats,
        **quantile_stats,
    }

    return stats_dict


def _process_datetime_column_data(column_data: dict) -> dict:
    column_number = column_data["column_number"]
    column_name = column_data["column_name"]
    column_type = column_data["column_type"]

    long_column_type = len(column_type) > 22

    if long_column_type:
        column_type_style = "font-size: 7.5px; color: gray; margin-top: 3px; margin-bottom: 2px;"
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
    stats_dict = {
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

    return stats_dict


def _process_boolean_column_data(column_data: dict) -> dict:
    column_number = column_data["column_number"]
    column_name = column_data["column_name"]
    column_type = column_data["column_type"]

    column_name_and_type = (
        f"<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>{column_name}</div>"
        f"<div style='font-size: 11px; color: gray;'>{column_type}</div>"
    )

    # Get the Missing and Unique value counts and fractions
    missing_vals = column_data["n_missing_values"]
    missing_vals_frac = _compact_decimal_fmt(column_data["f_missing_values"])
    missing_vals_str = f"{missing_vals}<br>{missing_vals_frac}"

    # Get the fractions of True and False values
    f_true_values = column_data["statistics"]["boolean"]["f_true_values"]
    f_false_values = column_data["statistics"]["boolean"]["f_false_values"]

    true_vals_frac_fmt = _compact_0_1_fmt(f_true_values)
    false_vals_frac_fmt = _compact_0_1_fmt(f_false_values)

    # Create an HTML string that combines fractions for the True and False values
    true_false_vals_str = f"<span style='font-weight: bold;'>T</span>{true_vals_frac_fmt}<br><span style='font-weight: bold;'>F</span>{false_vals_frac_fmt}"

    # unique_vals_str = f"{unique_vals}<br>{unique_vals_frac}"

    # Create a single dictionary with the statistics for the column
    stats_dict = {
        "column_number": column_number,
        "icon": SVG_ICONS_FOR_DATA_TYPES["boolean"],
        "column_name": column_name_and_type,
        "missing_vals": missing_vals_str,
        "unique_vals": true_false_vals_str,
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

    return stats_dict


def _process_other_column_data(column_data: dict) -> dict:
    raise
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
    stats_dict = {
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

    return stats_dict
