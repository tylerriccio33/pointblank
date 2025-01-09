from __future__ import annotations

from typing import Any

from narwhals.typing import FrameT
from great_tables import GT, style, loc, google_font, html

from pointblank.column import Column
from pointblank.schema import Schema
from pointblank._utils import _get_tbl_type, _check_any_df_lib, _select_df_lib

__all__ = ["preview"]


def preview(
    data: FrameT | Any,
    columns_subset: str | list[str] | Column | None = None,
    n_head: int = 5,
    n_tail: int = 5,
    limit: int | None = 50,
    show_row_numbers: bool = True,
    incl_header: bool = True,
    max_col_width: int | None = 250,
) -> GT:
    """
    Display a table preview that shows some rows from the top, some from the bottom.

    To get a quick look at the data in a table, we can use the `preview()` function to display a
    preview of the table. The function shows a subset of the rows from the start and end of the
    table, with the number of rows from the start and end determined by the `n_head=` and `n_tail=`
    parameters (set to `5` by default). This function works with any table that is supported by the
    `pointblank` library, including Pandas, Polars, and Ibis backend tables (e.g., DuckDB, MySQL,
    PostgreSQL, SQLite, Parquet, etc.).

    The view is optimized for readability, with column names and data types displayed in a compact
    format. The column widths are sized to fit the column names, dtypes, and column content up to
    a configurable maximum width of `max_col_width=` pixels. The table can be scrolled horizontally
    to view even very large datasets. Since the output is a Great Tables (`GT`) object, it can be
    further customized using the `great_tables` API.

    Parameters
    ----------
    data
        The table to preview, which could be a DataFrame object or an Ibis table object. Read the
        *Supported Input Table Types* section for details on the supported table types.
    columns_subset
        The columns to display in the table, by default `None` (all columns are shown). This can
        be a string, a list of strings, a `Column` object, or a `ColumnSelector` object. The latter
        two options allow for more flexible column selection using column selector functions. Errors
        are raised if the column names provided don't match any columns in the table (when provided
        as a string or list of strings) or if column selector expressions don't resolve to any
        columns.
    n_head
        The number of rows to show from the start of the table. Set to `5` by default.
    n_tail
        The number of rows to show from the end of the table. Set to `5` by default.
    limit
        The limit value for the sum of `n_head=` and `n_tail=` (the total number of rows shown).
        If the sum of `n_head=` and `n_tail=` exceeds the limit, an error is raised.
    show_row_numbers
        Should row numbers be shown? The numbers shown reflect the row numbers of the head and tail
        in the full table.
    incl_header
        Should the table include a header with the table type and table dimensions? Set to `True` by
        default.
    max_col_width
        The maximum width of the columns in pixels. This is `250` (`"250px"`) by default.

    Returns
    -------
    GT
        A GT object that displays the preview of the table.

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
    `ibis.expr.types.relations.Table`). Furthermore, using `preview()` with these types of tables
    requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or
    Pandas DataFrame, the availability of Ibis is not needed.

    Examples
    --------
    It's easy to preview a table using the `preview()` function. Here's an example using the
    `small_table` dataset (itself loaded using the `load_dataset()` function):

    ```{python}
    import pointblank as pb

    small_table_polars = pb.load_dataset("small_table")

    pb.preview(small_table_polars)
    ```

    This table is a Polars DataFrame, but the `preview()` function works with any table supported
    by `pointblank`, including Pandas DataFrames and Ibis backend tables. Here's an example using
    a DuckDB table handled by Ibis:

    ```{python}
    small_table_duckdb = pb.load_dataset("small_table", tbl_type="duckdb")

    pb.preview(small_table_duckdb)
    ```

    The blue dividing line marks the end of the first `n_head=` rows and the start of the last
    `n_tail=` rows.

    We can adjust the number of rows shown from the start and end of the table by setting the
    `n_head=` and `n_tail=` parameters. Let's enlarge each of these to `10`:

    ```{python}
    pb.preview(small_table_polars, n_head=10, n_tail=10)
    ```

    In the above case, the entire dataset is shown since the sum of `n_head=` and `n_tail=` is
    greater than the number of rows in the table (which is 13).

    The `columns_subset=` parameter can be used to show only specific columns in the table. You can
    provide a list of column names to make the selection. Let's try that with the `"game_revenue"`
    dataset as a Pandas DataFrame:

    ```{python}
    game_revenue_pandas = pb.load_dataset("game_revenue", tbl_type="pandas")

    pb.preview(game_revenue_pandas, columns_subset=["player_id", "item_name", "item_revenue"])
    ```

    Alternatively, we can use column selector functions like `starts_with()` and `matches()` to
    select columns based on text or patterns:

    ```{python}
    pb.preview(game_revenue_pandas, n_head=2, n_tail=2, columns_subset=pb.starts_with("item"))
    ```

    Multiple column selector functions can be combined within `col()` using operators like `|` and
    `&`:

    ```{python}
    pb.preview(
      game_revenue_pandas,
      n_head=2,
      n_tail=2,
      columns_subset=pb.col(pb.starts_with("item") | pb.matches("player"))
    )
    ```
    """

    # Check that the n_head and n_tail aren't greater than the limit
    if n_head + n_tail > limit:
        raise ValueError(f"The sum of `n_head=` and `n_tail=` cannot exceed the limit ({limit}).")

    # Do we have a DataFrame library to work with? We need at least one to display
    # the table using Great Tables
    _check_any_df_lib(method_used="preview_tbl")

    # Set flag for whether the full dataset is shown, or just the head and tail; if the table
    # is very small, the value likely will be `True`
    full_dataset = False

    # Determine if the table is a DataFrame or an Ibis table
    tbl_type = _get_tbl_type(data=data)
    ibis_tbl = "ibis.expr.types.relations.Table" in str(type(data))
    pl_pb_tbl = "polars" in tbl_type or "pandas" in tbl_type

    # Select the DataFrame library to use for displaying the Ibis table
    df_lib_gt = _select_df_lib(preference="polars")
    df_lib_name_gt = df_lib_gt.__name__

    # If the table is a DataFrame (Pandas or Polars), set `df_lib_name_gt` to the name of the
    # library (e.g., "polars" or "pandas")
    if pl_pb_tbl:
        df_lib_name_gt = "polars" if "polars" in tbl_type else "pandas"

        # Handle imports of Polars or Pandas here
        if df_lib_name_gt == "polars":
            import polars as pl
        else:
            import pandas as pd

    # If `columns_subset=` is not None, resolve the columns to display
    if columns_subset is not None:

        col_names = _get_column_names(data, ibis_tbl=ibis_tbl, df_lib_name_gt=df_lib_name_gt)

        resolved_columns = _validate_columns_subset(
            columns_subset=columns_subset, col_names=col_names
        )

        if len(resolved_columns) == 0:
            raise ValueError(
                "The `columns_subset=` value doesn't resolve to any columns in the table."
            )

        # Select the columns to display in the table with the `resolved_columns` value
        data = _select_columns(
            data, resolved_columns=resolved_columns, ibis_tbl=ibis_tbl, tbl_type=tbl_type
        )

    # From an Ibis table:
    # - get the row count
    # - subset the table to get the first and last n rows (if small, don't filter the table)
    # - get the row numbers for the table
    # - convert the table to a Polars or Pandas DF
    if ibis_tbl:

        # Get the Schema of the table
        tbl_schema = Schema(tbl=data)

        # Get the row count for the table
        ibis_rows = data.count()
        n_rows = ibis_rows.to_polars() if df_lib_name_gt == "polars" else int(ibis_rows.to_pandas())

        # If n_head + n_tail is greater than the row count, display the entire table
        if n_head + n_tail > n_rows:
            full_dataset = True
            data_subset = data
            row_number_list = range(1, n_rows + 1)
        else:
            # Get the first and last n rows of the table
            data_head = data.head(n=n_head)
            row_numbers_head = range(1, n_head + 1)
            data_tail = data[(n_rows - n_tail) : n_rows]
            row_numbers_tail = range(n_rows - n_tail + 1, n_rows + 1)
            data_subset = data_head.union(data_tail)
            row_number_list = list(row_numbers_head) + list(row_numbers_tail)

        # Convert either to Polars or Pandas depending on the available library
        if df_lib_name_gt == "polars":
            data = data_subset.to_polars()
        else:
            data = data_subset.to_pandas()

    # From a DataFrame:
    # - get the row count
    # - subset the table to get the first and last n rows (if small, don't filter the table)
    # - get the row numbers for the table
    if pl_pb_tbl:

        # Get the Schema of the table
        tbl_schema = Schema(tbl=data)

        if tbl_type == "polars":

            n_rows = int(data.height)

            # If n_head + n_tail is greater than the row count, display the entire table
            if n_head + n_tail >= n_rows:
                full_dataset = True
                row_number_list = range(1, n_rows + 1)
            else:
                data = pl.concat([data.head(n=n_head), data.tail(n=n_tail)])

                row_number_list = list(range(1, n_head + 1)) + list(
                    range(n_rows - n_tail + 1, n_rows + 1)
                )

        if tbl_type == "pandas":

            n_rows = data.shape[0]

            # If n_head + n_tail is greater than the row count, display the entire table
            if n_head + n_tail >= n_rows:
                full_dataset = True
                data_subset = data

                row_number_list = range(1, n_rows + 1)
            else:
                data = pd.concat([data.head(n=n_head), data.tail(n=n_tail)])

                row_number_list = list(range(1, n_head + 1)) + list(
                    range(n_rows - n_tail + 1, n_rows + 1)
                )

    # From the table schema, get a list of tuples containing column names and data types
    col_dtype_dict = tbl_schema.columns

    # Extract the column names from the list of tuples (first element of each tuple)
    col_names = [col[0] for col in col_dtype_dict]

    # Iterate over the list of tuples and create a new dictionary with the
    # column names and data types
    col_dtype_dict = {k: v for k, v in col_dtype_dict}

    # Create short versions of the data types by omitting any text in parentheses
    col_dtype_dict_short = {
        k: v.split("(")[0] if "(" in v else v for k, v in col_dtype_dict.items()
    }

    # Create a dictionary of column and row positions where the value is None/NA/NULL
    # This is used to highlight these values in the table
    if df_lib_name_gt == "polars":
        none_values = {k: data[k].is_null().to_list() for k in col_names}
    else:
        none_values = {k: data[k].isnull() for k in col_names}

    none_values = [(k, i) for k, v in none_values.items() for i, val in enumerate(v) if val]

    # Import Great Tables to get preliminary renders of the columns
    import great_tables as gt

    # For each of the columns get the average number of characters printed for each of the values
    max_length_col_vals = []

    for column in col_dtype_dict.keys():

        # Select a single column of values
        data_col = data[[column]] if df_lib_name_gt == "pandas" else data.select([column])

        # Using Great Tables, render the columns and get the list of values as formatted strings
        built_gt = GT(data=data_col).fmt_markdown(columns=column)._build_data(context="html")
        column_values = gt.gt._get_column_of_values(built_gt, column_name=column, context="html")

        # Get the maximum number of characters in the column
        max_length_col_vals.append(max([len(str(val)) for val in column_values]))

    length_col_names = [len(column) for column in col_dtype_dict.keys()]
    length_data_types = [len(dtype) for dtype in col_dtype_dict_short.values()]

    # Comparing the length of the column names, the data types, and the max length of the
    # column values, prefer the largest of these for the column widths (by column)
    col_widths = [
        f"{round(min(max(7.8 * max_length_col_vals[i] + 10, 7.8 * length_col_names[i] + 10, 7.8 * length_data_types[i] + 10), max_col_width))}px"
        for i in range(len(col_dtype_dict.keys()))
    ]

    # Set the column width to the col_widths list
    col_width_dict = {k: v for k, v in zip(col_names, col_widths)}

    # For each of the values in the dictionary, prepend the column name to the data type
    col_dtype_labels_dict = {
        k: html(
            f"<div><div style='white-space: nowrap; text-overflow: ellipsis; overflow: hidden; "
            f"padding-bottom: 2px; margin-bottom: 2px;'>{k}</div><div style='white-space: nowrap; "
            f"text-overflow: ellipsis; overflow: hidden; padding-top: 2px; margin-top: 2px;'>"
            f"<em>{v}</em></div></div>"
        )
        for k, v in col_dtype_dict_short.items()
    }

    # Prepend a column that contains the row numbers if `show_row_numbers=True`
    if show_row_numbers:

        if df_lib_name_gt == "polars":

            import polars as pl

            row_number_series = pl.Series("_row_num_", row_number_list)
            data = data.insert_column(0, row_number_series)

        if df_lib_name_gt == "pandas":

            data.insert(0, "_row_num_", row_number_list)

        # Get the highest number in the `row_number_list` and calculate a width that will
        # safely fit a number of that magnitude
        max_row_num = max(row_number_list)
        max_row_num_width = len(str(max_row_num)) * 7.8 + 10

        # Update the col_width_dict to include the row number column
        col_width_dict = {"_row_num_": f"{max_row_num_width}px"} | col_width_dict
        # Update the col_dtype_labels_dict to include the row number column (use empty string)
        col_dtype_labels_dict = {"_row_num_": ""} | col_dtype_labels_dict

    gt_tbl = (
        GT(data=data, id="pb_preview_tbl")
        .opt_table_font(font=google_font(name="IBM Plex Sans"))
        .fmt_markdown(columns=col_names)
        .tab_style(
            style=style.css(
                "height: 14px; padding: 4px; white-space: nowrap; text-overflow: "
                "ellipsis; overflow: hidden;"
            ),
            locations=loc.body(),
        )
        .tab_style(
            style=style.text(color="black", font=google_font(name="IBM Plex Mono"), size="12px"),
            locations=loc.body(),
        )
        .tab_style(
            style=style.text(color="gray20", font=google_font(name="IBM Plex Mono"), size="12px"),
            locations=loc.column_labels(),
        )
        .tab_style(
            style=style.borders(
                sides=["top", "bottom"], color="#E9E9E", style="solid", weight="1px"
            ),
            locations=loc.body(),
        )
        .tab_options(
            table_body_vlines_style="solid",
            table_body_vlines_width="1px",
            table_body_vlines_color="#E9E9E9",
            column_labels_vlines_style="solid",
            column_labels_vlines_width="1px",
            column_labels_vlines_color="#F2F2F2",
        )
        .cols_label(cases=col_dtype_labels_dict)
        .cols_width(cases=col_width_dict)
    )

    if none_values:
        for column, none_index in none_values:
            gt_tbl = gt_tbl.tab_style(
                style=[style.text(color="#B22222"), style.fill(color="#FFC1C159")],
                locations=loc.body(rows=none_index, columns=column),
            )

        if tbl_type == "pandas":
            gt_tbl = gt_tbl.sub_missing(missing_text="NA")

        if ibis_tbl:
            gt_tbl = gt_tbl.sub_missing(missing_text="NULL")

    if not full_dataset:

        gt_tbl = gt_tbl.tab_style(
            style=style.borders(sides="bottom", color="#6699CC80", style="solid", weight="2px"),
            locations=loc.body(rows=n_head - 1),
        )

    if show_row_numbers:

        gt_tbl = gt_tbl.tab_style(
            style=[
                style.text(color="gray", font=google_font(name="IBM Plex Mono"), size="10px"),
                style.borders(sides="right", color="#6699CC80", style="solid", weight="2px"),
            ],
            locations=loc.body(columns="_row_num_"),
        )

    return gt_tbl


def _get_column_names(data: FrameT | Any, ibis_tbl: bool, df_lib_name_gt: str) -> list[str]:
    if ibis_tbl:
        return data.columns if df_lib_name_gt == "polars" else list(data.columns)
    return list(data.columns)


def _validate_columns_subset(
    columns_subset: str | list[str] | Column, col_names: list[str]
) -> list[str]:
    if isinstance(columns_subset, str):
        if columns_subset not in col_names:
            raise ValueError("The `columns_subset=` value doesn't match any columns in the table.")
        return [columns_subset]

    if isinstance(columns_subset, list):
        if all(isinstance(col, str) for col in columns_subset):
            if not all(col in col_names for col in columns_subset):
                raise ValueError(
                    "Not all columns provided as `columns_subset=` match the table's columns."
                )
            return columns_subset

    return columns_subset.resolve(columns=col_names)


def _select_columns(
    data: FrameT | Any, resolved_columns: list[str], ibis_tbl: bool, tbl_type: str
) -> FrameT | Any:
    if ibis_tbl:
        return data[resolved_columns]
    if tbl_type == "polars":
        return data.select(resolved_columns)
    return data[resolved_columns]
