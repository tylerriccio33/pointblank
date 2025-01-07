from __future__ import annotations

from typing import Any

from narwhals.typing import FrameT
from great_tables import GT, style, loc, google_font, html

from pointblank.schema import Schema
from pointblank._utils import _get_tbl_type, _check_any_df_lib, _select_df_lib

__all__ = ["preview"]


def preview(
    data: FrameT | Any,
    n_head: int = 5,
    n_tail: int = 5,
    limit: int | None = 50,
) -> GT:

    # Check that the n_head and n_tail aren't greater than the limit
    if n_head + n_tail > limit:
        raise ValueError(f"The sum of `n_head=` and `n_tail=` cannot exceed the limit ({limit}).")

    # Do we have a DataFrame library to work with?
    _check_any_df_lib(method_used="preview_tbl")

    # Select the DataFrame library to use for viewing the table
    df_lib = _select_df_lib(preference="polars")
    df_lib_name = df_lib.__name__

    # Determine if the table is a DataFrame or a DB table
    tbl_type = _get_tbl_type(data=data)

    ibis_tbl = "ibis.expr.types.relations.Table" in str(type(data))
    pl_pb_tbl = "polars" in tbl_type or "pandas" in tbl_type

    if ibis_tbl:

        # Get the row count for the table
        ibis_rows = data.count()
        n_rows = ibis_rows.to_polars() if df_lib_name == "polars" else int(ibis_rows.to_pandas())

        # If n_head + n_tail is greater than the row count, display the entire table
        if n_head + n_tail > n_rows:
            data_subset = data
        else:
            # Get the first and last n rows of the table
            data_head = data.head(n=n_head)
            data_tail = data[(n_rows - n_tail) : n_rows]
            data_subset = data_head.union(data_tail)

        # Convert to Polars DF
        if df_lib_name == "pandas":
            data = data_subset.to_pandas()
        else:
            data = data_subset.to_polars()

    if pl_pb_tbl:

        if tbl_type == "polars":

            import polars as pl

            n_rows = int(data.height)

            # If n_head + n_tail is greater than the row count, display the entire table
            if n_head + n_tail > n_rows:
                data_subset = data
            else:
                data = pl.concat([data.head(n=n_head), data.tail(n=n_tail)])

        if tbl_type == "pandas":

            import pandas as pd

            n_rows = data.shape[0]

            # If n_head + n_tail is greater than the row count, display the entire table
            if n_head + n_tail > n_rows:
                data_subset = data
            else:
                data = pd.concat([data.head(n=n_head), data.tail(n=n_tail)])

    # Get the Schema of the table
    tbl_schema = Schema(tbl=data)

    # Get dictionary of column names and data types
    col_dtype_dict = tbl_schema.columns

    # Get a list of column names
    if ibis_tbl:
        col_names = data.columns if df_lib_name == "polars" else list(data.columns)
    else:
        col_names = list(data.columns)

    # Iterate over the list of tuples and create a new dictionary with the
    # column names and data types
    col_dtype_dict = {k: v for k, v in col_dtype_dict}

    # Create short versions of the data types by omitting any text in parentheses
    col_dtype_dict_short = {
        k: v.split("(")[0] if "(" in v else v for k, v in col_dtype_dict.items()
    }

    import great_tables as gt

    # For each of the columns get the average number of characters printed for each of the values
    max_length_col_vals = []

    for col in col_dtype_dict.keys():

        if ibis_tbl:
            if df_lib_name == "pandas":
                data_col = data[[col]]
            else:
                data_col = data.select([col])

        else:
            if tbl_type == "polars":
                data_col = data.select([col])
            else:
                data_col = data[[col]]

        built_gt = GT(data=data_col).fmt_markdown(columns=col)._build_data(context="html")
        column_values = gt.gt._get_column_of_values(built_gt, column_name=col, context="html")

        # Get the maximum number of characters in the column
        max_length_col_vals.append(max([len(str(val)) for val in column_values]))

    length_col_names = [len(col) for col in col_dtype_dict.keys()]
    length_data_types = [len(dtype) for dtype in col_dtype_dict_short.values()]

    # Comparing the length of the column names, the data types, and the max length of the
    # column values, prefer the largest of these for the column widths (by column)
    col_widths = [
        f"{max(7.8 * max_length_col_vals[i] + 10, 7.8 * length_col_names[i] + 10, 7.8 * length_data_types[i] + 10)}px"
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
                sides=["top", "bottom"], color="#E5E5E5", style="dashed", weight="1px"
            ),
            locations=loc.body(),
        )
        .cols_label(cases=col_dtype_labels_dict)
        .cols_width(cases=col_width_dict)
    )

    gt_tbl = gt_tbl.tab_style(
        style=style.borders(sides="bottom", color="#6699CC80", style="solid", weight="2px"),
        locations=loc.body(rows=n_head - 1),
    )

    return gt_tbl
