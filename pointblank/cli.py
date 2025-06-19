from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import pointblank as pb
from pointblank._utils import _get_tbl_type, _is_lib_present

console = Console()


def _format_cell_value(
    value: Any, is_row_number: bool = False, max_width: int = 50, num_columns: int = 10
) -> str:
    """Format a cell value for Rich table display, highlighting None/NA values in red.

    Args:
        value: The raw cell value from the dataframe
        is_row_number: Whether this is a row number column value
        max_width: Maximum character width for text truncation
        num_columns: Number of columns in the table (affects truncation aggressiveness)

    Returns:
        Formatted string with Rich markup for None/NA values or row numbers
    """
    # Special formatting for row numbers: never truncate them
    if is_row_number:
        return f"[dim]{value}[/dim]"

    # Check for actual None/null values (not string representations)
    if value is None:
        return "[red]None[/red]"

    # Check for pandas/numpy specific NA values
    try:
        import numpy as np
        import pandas as pd

        # Check for pandas NA
        if pd.isna(value):
            # If it's specifically numpy.nan, show as NaN
            if isinstance(value, float) and np.isnan(value):
                return "[red]NaN[/red]"
            # If it's pandas NA, show as NA
            elif str(type(value)).find("pandas") != -1:
                return "[red]NA[/red]"
            # Generic NA for other pandas missing values
            else:
                return "[red]NA[/red]"

    except (ImportError, TypeError):
        # If pandas/numpy not available or value not compatible, continue
        pass

    # Check for empty strings (but only actual empty strings, not whitespace)
    if isinstance(value, str) and value == "":
        return "[red][/red]"  # Empty string shown as red empty space

    # Convert to string and apply intelligent truncation
    str_value = str(value)

    # Adjust max_width based on number of columns to prevent overly wide tables
    if num_columns > 15:
        adjusted_max_width = min(max_width, 30)  # Be more aggressive with many columns
    elif num_columns > 10:
        adjusted_max_width = min(max_width, 40)
    else:
        adjusted_max_width = max_width

    # Apply truncation if the string is too long
    if len(str_value) > adjusted_max_width:
        # For very long text, truncate more aggressively
        if len(str_value) > adjusted_max_width * 2:
            # For extremely long text, use a shorter truncation
            truncated = str_value[: adjusted_max_width // 2] + "…"
        else:
            # For moderately long text, use a more generous truncation
            truncated = str_value[: adjusted_max_width - 1] + "…"

        return truncated

    return str_value


def _get_column_dtypes(df: Any, columns: list[str]) -> dict[str, str]:
    """Extract data types for columns and format them in a compact way.

    Args:
        df: The dataframe object
        columns: List of column names

    Returns:
        Dictionary mapping column names to formatted data type strings
    """
    dtypes_dict = {}

    try:
        if hasattr(df, "dtypes"):
            # Polars/Pandas style
            if hasattr(df.dtypes, "to_dict"):
                # Polars DataFrame dtypes
                raw_dtypes = df.dtypes.to_dict() if hasattr(df.dtypes, "to_dict") else {}
                for col in columns:
                    if col in raw_dtypes:
                        dtype_str = str(raw_dtypes[col])
                        # Convert to compact format similar to Polars glimpse()
                        dtypes_dict[col] = _format_dtype_compact(dtype_str)
                    else:
                        dtypes_dict[col] = "?"
            else:
                # Pandas DataFrame dtypes (Series-like)
                for i, col in enumerate(columns):
                    if i < len(df.dtypes):
                        dtype_str = str(
                            df.dtypes.iloc[i] if hasattr(df.dtypes, "iloc") else df.dtypes[i]
                        )
                        dtypes_dict[col] = _format_dtype_compact(dtype_str)
                    else:
                        dtypes_dict[col] = "?"
        elif hasattr(df, "schema"):
            # Other schema-based systems (e.g., Ibis)
            schema = df.schema
            if hasattr(schema, "to_dict"):
                raw_dtypes = schema.to_dict()
                for col in columns:
                    if col in raw_dtypes:
                        dtypes_dict[col] = _format_dtype_compact(str(raw_dtypes[col]))
                    else:
                        dtypes_dict[col] = "?"
            else:
                for col in columns:
                    try:
                        dtype_str = str(getattr(schema, col, "Unknown"))
                        dtypes_dict[col] = _format_dtype_compact(dtype_str)
                    except Exception:
                        dtypes_dict[col] = "?"
        else:
            # Fallback: no type information available
            for col in columns:
                dtypes_dict[col] = "?"

    except Exception:
        # If any error occurs, fall back to unknown types
        for col in columns:
            dtypes_dict[col] = "?"

    return dtypes_dict


def _format_dtype_compact(dtype_str: str) -> str:
    """Format a data type string to a compact representation.

    Args:
        dtype_str: The raw data type string

    Returns:
        Compact formatted data type string
    """
    # Remove common prefixes and make compact
    dtype_str = dtype_str.lower()

    # Polars types
    if "utf8" in dtype_str or "string" in dtype_str:
        return "str"
    elif "int64" in dtype_str:
        return "i64"
    elif "int32" in dtype_str:
        return "i32"
    elif "float64" in dtype_str:
        return "f64"
    elif "float32" in dtype_str:
        return "f32"
    elif "boolean" in dtype_str or "bool" in dtype_str:
        return "bool"
    elif "datetime" in dtype_str:
        return "datetime"
    elif "date" in dtype_str and "datetime" not in dtype_str:
        return "date"
    elif "time" in dtype_str:
        return "time"

    # Pandas types
    elif "object" in dtype_str:
        return "obj"
    elif "category" in dtype_str:
        return "cat"

    # Generic fallbacks
    elif "int" in dtype_str:
        return "int"
    elif "float" in dtype_str:
        return "float"
    elif "str" in dtype_str:
        return "str"

    # Unknown or complex types - truncate if too long
    elif len(dtype_str) > 8:
        return dtype_str[:8] + "…"
    else:
        return dtype_str


def _rich_print_gt_table(gt_table: Any, preview_info: dict | None = None) -> None:
    """Convert a GT table to Rich table and display it in the terminal.

    Args:
        gt_table: The GT table object to display
        preview_info: Optional dict with preview context info:
            - total_rows: Total rows in the dataset
            - head_rows: Number of head rows shown
            - tail_rows: Number of tail rows shown
            - is_complete: Whether the entire dataset is shown
    """
    try:
        # Try to extract the underlying data from the GT table
        df = None

        # Great Tables stores the original data in different places depending on how it was created
        # Let's try multiple approaches to get the data
        if hasattr(gt_table, "_tbl_data") and gt_table._tbl_data is not None:
            df = gt_table._tbl_data
        elif (
            hasattr(gt_table, "_body")
            and hasattr(gt_table._body, "body")
            and gt_table._body.body is not None
        ):
            df = gt_table._body.body
        elif hasattr(gt_table, "_data") and gt_table._data is not None:
            df = gt_table._data
        elif hasattr(gt_table, "data") and gt_table.data is not None:
            df = gt_table.data

        if df is not None:
            # Create a Rich table with horizontal lines
            from rich.box import SIMPLE_HEAD

            # Create enhanced title if preview_info contains metadata
            table_title = None
            if preview_info and "source_type" in preview_info and "table_type" in preview_info:
                source_type = preview_info["source_type"]
                table_type = preview_info["table_type"]
                table_title = f"Data Preview / {source_type} / {table_type}"

            rich_table = Table(
                title=table_title,
                show_header=True,
                header_style="bold magenta",
                box=SIMPLE_HEAD,
                title_style="bold cyan",
                title_justify="left",
            )

            # Get column names
            columns = []
            if hasattr(df, "columns"):
                columns = list(df.columns)
            elif hasattr(df, "schema"):
                columns = list(df.schema.names)
            elif hasattr(df, "column_names"):
                columns = list(df.column_names)

            if not columns:
                # Fallback: try to determine columns from first row
                try:
                    if hasattr(df, "to_dicts") and len(df) > 0:
                        first_dict = df.to_dicts()[0]
                        columns = list(first_dict.keys())
                    elif hasattr(df, "to_dict") and len(df) > 0:
                        first_dict = df.to_dict("records")[0]
                        columns = list(first_dict.keys())
                except Exception:
                    columns = [f"Column {i + 1}" for i in range(10)]  # Default fallback

            # Add columns to Rich table
            # Handle wide tables by limiting columns displayed
            max_terminal_cols = 15  # Reasonable limit for terminal display

            # Get terminal width to adjust column behavior
            try:
                terminal_width = console.size.width
                # Estimate max column width based on terminal size and number of columns
                if len(columns) <= 5:
                    max_col_width = min(60, terminal_width // 4)
                elif len(columns) <= 10:
                    max_col_width = min(40, terminal_width // 6)
                else:
                    max_col_width = min(30, terminal_width // 8)
            except Exception:
                # Fallback if we can't get terminal width
                max_col_width = 40 if len(columns) <= 10 else 25

            if len(columns) > max_terminal_cols:
                # For wide tables, show first few, middle indicator, and last few columns
                first_cols = 7
                last_cols = 7

                display_columns = columns[:first_cols] + ["...more..."] + columns[-last_cols:]

                console.print(
                    f"\n[yellow]⚠ Table has {len(columns)} columns. Showing first {first_cols} and last {last_cols} columns.[/yellow]"
                )
                console.print("[dim]Use --columns to specify which columns to display.[/dim]")
                console.print(
                    f"[dim]Full column list: {', '.join(columns[:5])}...{', '.join(columns[-5:])}[/dim]\n"
                )
            else:
                display_columns = columns

            # Get data types for columns
            dtypes_dict = _get_column_dtypes(df, columns)

            # Calculate row number column width if needed
            row_num_width = 6  # Default width
            if "_row_num_" in columns:
                try:
                    # Get the maximum row number to calculate appropriate width
                    if hasattr(df, "to_dicts"):
                        data_dict = df.to_dicts()
                        if data_dict:
                            row_nums = [row.get("_row_num_", 0) for row in data_dict]
                            max_row_num = max(row_nums) if row_nums else 0
                            row_num_width = max(len(str(max_row_num)) + 1, 6)  # +1 for padding
                    elif hasattr(df, "to_dict"):
                        data_dict = df.to_dict("records")
                        if data_dict:
                            row_nums = [row.get("_row_num_", 0) for row in data_dict]
                            max_row_num = max(row_nums) if row_nums else 0
                            row_num_width = max(len(str(max_row_num)) + 1, 6)  # +1 for padding
                except Exception:
                    # If we can't determine max row number, use default
                    row_num_width = 8  # Slightly larger default for safety

            for i, col in enumerate(display_columns):
                if col == "...more...":
                    # Add a special indicator column
                    rich_table.add_column("···", style="dim", width=3, no_wrap=True)
                else:
                    # Handle row number column specially
                    if col == "_row_num_":
                        # Row numbers get no header, right alignment, and dim gray style
                        # Use dynamic width to prevent truncation
                        rich_table.add_column(
                            "", style="dim", justify="right", no_wrap=True, width=row_num_width
                        )
                    else:
                        display_col = str(col)

                        # Get data type for this column (if available)
                        if col in dtypes_dict:
                            dtype_display = f"<{dtypes_dict[col]}>"
                            # Create header with column name and data type
                            header_text = f"{display_col}\n[dim yellow]{dtype_display}[/dim yellow]"
                        else:
                            header_text = display_col

                        rich_table.add_column(
                            header_text,
                            style="cyan",
                            no_wrap=False,
                            overflow="ellipsis",
                            max_width=max_col_width,
                        )

            # Convert data to list of rows
            rows = []
            try:
                if hasattr(df, "to_dicts"):
                    # Polars interface
                    data_dict = df.to_dicts()
                    if len(columns) > max_terminal_cols:
                        # For wide tables, extract only the displayed columns
                        display_data_columns = (
                            columns[:7] + columns[-7:]
                        )  # Skip the "...more..." placeholder
                        rows = [
                            [
                                _format_cell_value(
                                    row.get(col, ""),
                                    is_row_number=(col == "_row_num_"),
                                    max_width=max_col_width,
                                    num_columns=len(columns),
                                )
                                for col in display_data_columns
                            ]
                            for row in data_dict
                        ]
                        # Add the "..." column in the middle
                        for i, row in enumerate(rows):
                            rows[i] = row[:7] + ["···"] + row[7:]
                    else:
                        rows = [
                            [
                                _format_cell_value(
                                    row.get(col, ""),
                                    is_row_number=(col == "_row_num_"),
                                    max_width=max_col_width,
                                    num_columns=len(columns),
                                )
                                for col in columns
                            ]
                            for row in data_dict
                        ]
                elif hasattr(df, "to_dict"):
                    # Pandas-like interface
                    data_dict = df.to_dict("records")
                    if len(columns) > max_terminal_cols:
                        # For wide tables, extract only the displayed columns
                        display_data_columns = columns[:7] + columns[-7:]
                        rows = [
                            [
                                _format_cell_value(
                                    row.get(col, ""),
                                    is_row_number=(col == "_row_num_"),
                                    max_width=max_col_width,
                                    num_columns=len(columns),
                                )
                                for col in display_data_columns
                            ]
                            for row in data_dict
                        ]
                        # Add the "..." column in the middle
                        for i, row in enumerate(rows):
                            rows[i] = row[:7] + ["···"] + row[7:]
                    else:
                        rows = [
                            [
                                _format_cell_value(
                                    row.get(col, ""),
                                    is_row_number=(col == "_row_num_"),
                                    max_width=max_col_width,
                                    num_columns=len(columns),
                                )
                                for col in columns
                            ]
                            for row in data_dict
                        ]
                elif hasattr(df, "iter_rows"):
                    # Polars lazy frame
                    rows = [
                        [
                            _format_cell_value(
                                val,
                                is_row_number=(i == 0 and columns[0] == "_row_num_"),
                                max_width=max_col_width,
                                num_columns=len(columns),
                            )
                            for i, val in enumerate(row)
                        ]
                        for row in df.iter_rows()
                    ]
                elif hasattr(df, "__iter__"):
                    # Try to iterate directly
                    rows = [
                        [
                            _format_cell_value(
                                val,
                                is_row_number=(i == 0 and columns[0] == "_row_num_"),
                                max_width=max_col_width,
                                num_columns=len(columns),
                            )
                            for i, val in enumerate(row)
                        ]
                        for row in df
                    ]
                else:
                    rows = [["Could not extract data from this format"]]
            except Exception as e:
                rows = [[f"Error extracting data: {e}"]]

            # Add rows to Rich table with separator between head and tail
            max_rows = 50  # Reasonable limit for terminal display

            # Get preview info to determine head/tail separation
            head_rows_count = 0
            tail_rows_count = 0
            total_dataset_rows = 0

            if preview_info:
                head_rows_count = preview_info.get("head_rows", 0)
                tail_rows_count = preview_info.get("tail_rows", 0)
                total_dataset_rows = preview_info.get("total_rows", len(rows))
                is_complete = preview_info.get("is_complete", False)
            else:
                # Fallback: assume all rows are shown
                is_complete = True

            # Add rows with optional separator
            for i, row in enumerate(rows[:max_rows]):
                try:
                    # Add separator between head and tail rows
                    if (
                        not is_complete
                        and head_rows_count > 0
                        and tail_rows_count > 0
                        and i == head_rows_count
                    ):
                        # Add a visual separator row with dashes
                        separator_row = [
                            "─" * 3 if col != "_row_num_" else "⋮"
                            for col in (
                                display_columns if "display_columns" in locals() else columns
                            )
                        ]
                        rich_table.add_row(*separator_row, style="dim")

                    rich_table.add_row(*row)
                except Exception as e:
                    # If there's an issue with row data, show error
                    rich_table.add_row(*[f"Error: {e}" for _ in columns])
                    break

            # Show the table
            console.print()
            console.print(rich_table)

            # Show summary info
            total_rows = len(rows)

            # Use preview info if available, otherwise fall back to old logic
            if preview_info:
                total_dataset_rows = preview_info.get("total_rows", total_rows)
                head_rows = preview_info.get("head_rows", 0)
                tail_rows = preview_info.get("tail_rows", 0)
                is_complete = preview_info.get("is_complete", False)

                if is_complete:
                    console.print(f"\n[dim]Showing all {total_rows} rows.[/dim]")
                elif head_rows > 0 and tail_rows > 0:
                    console.print(
                        f"\n[dim]Showing first {head_rows} and last {tail_rows} rows from {total_dataset_rows:,} total rows.[/dim]"
                    )
                elif head_rows > 0:
                    console.print(
                        f"\n[dim]Showing first {head_rows} rows from {total_dataset_rows:,} total rows.[/dim]"
                    )
                elif tail_rows > 0:
                    console.print(
                        f"\n[dim]Showing last {tail_rows} rows from {total_dataset_rows:,} total rows.[/dim]"
                    )
                else:
                    # Fallback for other cases
                    console.print(
                        f"\n[dim]Showing {total_rows} rows from {total_dataset_rows:,} total rows.[/dim]"
                    )
            else:
                # Original logic as fallback
                max_rows = 50  # This should match the limit used above
                if total_rows > max_rows:
                    console.print(
                        f"\n[dim]Showing first {max_rows} of {total_rows} rows. Use --output-html to see all data.[/dim]"
                    )
                else:
                    console.print(f"\n[dim]Showing all {total_rows} rows.[/dim]")

        else:
            # If we can't extract data, show the success message
            console.print(
                Panel(
                    "[green]✓[/green] Table rendered successfully. "
                    "Use --output-html to save the full interactive report.",
                    title="Table Preview",
                    border_style="green",
                )
            )

    except Exception as e:
        console.print(f"[red]Error rendering table:[/red] {e}")
        console.print(
            f"[dim]GT table type: {type(gt_table) if 'gt_table' in locals() else 'undefined'}[/dim]"
        )

        # Fallback: show the success message
        console.print(
            Panel(
                "[green]✓[/green] Table rendered successfully. "
                "Use --output-html to save the full interactive report.",
                title="Table Preview",
                border_style="green",
            )
        )


def _display_validation_summary(validation: Any) -> None:
    """Display a validation summary in a Rich table format."""
    try:
        # Try to get the summary from the validation report
        if hasattr(validation, "validation_info") and validation.validation_info is not None:
            # Use the validation_info to create a summary
            info = validation.validation_info
            n_steps = len(info)
            n_passed = sum(1 for step in info if step.all_passed)
            n_failed = n_steps - n_passed

            # Calculate severity counts
            n_warning = sum(1 for step in info if step.warning)
            n_error = sum(1 for step in info if step.error)
            n_critical = sum(1 for step in info if step.critical)

            all_passed = n_failed == 0

            # Determine highest severity
            if n_critical > 0:
                highest_severity = "critical"
            elif n_error > 0:
                highest_severity = "error"
            elif n_warning > 0:
                highest_severity = "warning"
            elif n_failed > 0:
                highest_severity = "some failing"
            else:
                highest_severity = "all passed"

            # Create a summary table
            table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")

            # Add summary statistics
            table.add_row("Total Steps", str(n_steps))
            table.add_row("Passing Steps", str(n_passed))
            table.add_row("Failing Steps", str(n_failed))
            table.add_row("Warning Steps", str(n_warning))
            table.add_row("Error Steps", str(n_error))
            table.add_row("Critical Steps", str(n_critical))
            table.add_row("All Passed", str(all_passed))
            table.add_row("Highest Severity", highest_severity)

            console.print(table)

            # Display step details
            if n_steps > 0:
                steps_table = Table(
                    title="Validation Steps", show_header=True, header_style="bold cyan"
                )
                steps_table.add_column("Step", style="dim")
                steps_table.add_column("Type", style="white")
                steps_table.add_column("Column", style="cyan")
                steps_table.add_column("Status", style="white")
                steps_table.add_column("Passed/Total", style="green")

                for step in info:
                    status_icon = "✓" if step.all_passed else "✗"
                    status_color = "green" if step.all_passed else "red"

                    severity = ""
                    if step.critical:
                        severity = " [red](CRITICAL)[/red]"
                    elif step.error:
                        severity = " [red](ERROR)[/red]"
                    elif step.warning:
                        severity = " [yellow](WARNING)[/yellow]"

                    steps_table.add_row(
                        str(step.i),
                        step.assertion_type,
                        str(step.column) if step.column else "—",
                        f"[{status_color}]{status_icon}[/{status_color}]{severity}",
                        f"{step.n_passed}/{step.n}",
                    )

                console.print(steps_table)

            # Display status with appropriate color
            if highest_severity == "all passed":
                console.print(
                    Panel("[green]✓ All validations passed![/green]", border_style="green")
                )
            elif highest_severity == "some failing":
                console.print(
                    Panel("[yellow]⚠ Some validations failed[/yellow]", border_style="yellow")
                )
            elif highest_severity in ["warning", "error", "critical"]:
                color = "yellow" if highest_severity == "warning" else "red"
                console.print(
                    Panel(
                        f"[{color}]✗ Validation failed with {highest_severity} severity[/{color}]",
                        border_style=color,
                    )
                )
        else:
            console.print("[yellow]Validation object does not contain validation results.[/yellow]")

    except Exception as e:
        console.print(f"[red]Error displaying validation summary:[/red] {e}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@click.group()
@click.version_option(version=pb.__version__, prog_name="pb")
def cli():
    """
    Pointblank CLI - Data validation and quality tools for data engineers.

    Use this CLI to validate data, preview tables, and generate reports
    directly from the command line.
    """
    pass


@cli.command()
def datasets():
    """
    List available built-in datasets.
    """
    datasets_info = [
        ("small_table", "13 rows × 8 columns", "Small demo dataset for testing"),
        ("game_revenue", "2,000 rows × 11 columns", "Game development company revenue data"),
        ("nycflights", "336,776 rows × 18 columns", "NYC airport flights data from 2013"),
        ("global_sales", "50,000 rows × 20 columns", "Global sales data across regions"),
    ]

    table = Table(
        title="Available Pointblank Datasets", show_header=True, header_style="bold magenta"
    )
    table.add_column("Dataset Name", style="cyan", no_wrap=True)
    table.add_column("Dimensions", style="green")
    table.add_column("Description", style="white")

    for name, dims, desc in datasets_info:
        table.add_row(name, dims, desc)

    console.print(table)
    console.print("\n[dim]Use these dataset names directly with any pb CLI command.[/dim]")
    console.print("[dim]Example: pb preview small_table[/dim]")


@cli.command()
def requirements():
    """
    Check installed dependencies and their availability.
    """
    dependencies = [
        ("polars", "Polars DataFrame support"),
        ("pandas", "Pandas DataFrame support"),
        ("ibis", "Ibis backend support (DuckDB, etc.)"),
        ("duckdb", "DuckDB database support"),
        ("pyarrow", "Parquet file support"),
    ]

    table = Table(title="Dependency Status", show_header=True, header_style="bold magenta")
    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_column("Description", style="dim")

    for package, description in dependencies:
        if _is_lib_present(package):
            status = "[green]✓ Installed[/green]"
        else:
            status = "[red]✗ Not installed[/red]"

        table.add_row(package, status, description)

    console.print(table)
    console.print("\n[dim]Install missing packages to enable additional functionality.[/dim]")


@cli.command()
@click.argument("data_source", type=str)
@click.option("--columns", "-c", help="Comma-separated list of columns to display")
@click.option("--col-range", help="Column range like '1:10' or '5:' or ':15' (1-based indexing)")
@click.option("--col-first", type=int, help="Show first N columns")
@click.option("--col-last", type=int, help="Show last N columns")
@click.option("--head", "-h", default=5, help="Number of rows from the top (default: 5)")
@click.option("--tail", "-t", default=5, help="Number of rows from the bottom (default: 5)")
@click.option("--limit", "-l", default=50, help="Maximum total rows to display (default: 50)")
@click.option("--no-row-numbers", is_flag=True, help="Hide row numbers")
@click.option("--max-col-width", default=250, help="Maximum column width in pixels (default: 250)")
@click.option("--min-table-width", default=500, help="Minimum table width in pixels (default: 500)")
@click.option("--no-header", is_flag=True, help="Hide table header")
@click.option("--output-html", type=click.Path(), help="Save HTML output to file")
def preview(
    data_source: str,
    columns: str | None,
    col_range: str | None,
    col_first: int | None,
    col_last: int | None,
    head: int,
    tail: int,
    limit: int,
    no_row_numbers: bool,
    max_col_width: int,
    min_table_width: int,
    no_header: bool,
    output_html: str | None,
):
    """
    Preview a data table showing head and tail rows.

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

    COLUMN SELECTION OPTIONS:

    For tables with many columns, use these options to control which columns are displayed:

    \b
    - --columns: Specify exact columns (e.g., --columns "name,age,email")
    - --col-range: Select column range (e.g., --col-range "1:10", --col-range "5:", --col-range ":15")
    - --col-first: Show first N columns (e.g., --col-first 5)
    - --col-last: Show last N columns (e.g., --col-last 3)

    Tables with >15 columns automatically show first 7 and last 7 columns with indicators.
    """
    try:
        with console.status("[bold green]Loading data..."):
            # Try to load as a pointblank dataset first
            if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                data = pb.load_dataset(data_source)
                console.print(f"[green]✓[/green] Loaded dataset: {data_source}")
            else:
                # Assume it's a file path or connection string
                data = data_source
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Parse columns if provided
        columns_list = None
        if columns:
            columns_list = [col.strip() for col in columns.split(",")]

            # If data has _row_num_ and it's not explicitly included, add it at the beginning
            try:
                from pointblank.validate import (
                    _process_connection_string,
                    _process_csv_input,
                    _process_parquet_input,
                )

                # Process the data source to get actual data object to check for _row_num_
                processed_data = data
                if isinstance(data, str):
                    processed_data = _process_connection_string(data)
                    processed_data = _process_csv_input(processed_data)
                    processed_data = _process_parquet_input(processed_data)

                # Get column names from the processed data
                all_columns = []
                if hasattr(processed_data, "columns"):
                    all_columns = list(processed_data.columns)
                elif hasattr(processed_data, "schema"):
                    all_columns = list(processed_data.schema.names)

                # If _row_num_ exists in data but not in user selection, add it at beginning
                if all_columns and "_row_num_" in all_columns and "_row_num_" not in columns_list:
                    columns_list = ["_row_num_"] + columns_list
            except Exception:
                # If we can't process the data, just use the user's column list as-is
                pass
        elif col_range or col_first or col_last:
            # Need to get column names to apply range/first/last selection
            # Load the data to get column names
            from pointblank.validate import (
                _process_connection_string,
                _process_csv_input,
                _process_parquet_input,
            )

            # Process the data source to get actual data object
            processed_data = data
            if isinstance(data, str):
                processed_data = _process_connection_string(data)
                processed_data = _process_csv_input(processed_data)
                processed_data = _process_parquet_input(processed_data)

            # Get column names from the processed data
            all_columns = []
            if hasattr(processed_data, "columns"):
                all_columns = list(processed_data.columns)
            elif hasattr(processed_data, "schema"):
                all_columns = list(processed_data.schema.names)
            else:
                console.print(
                    "[yellow]Warning: Could not determine column names for range selection[/yellow]"
                )

            if all_columns:
                # Check if _row_num_ exists and preserve it
                has_row_num = "_row_num_" in all_columns

                if col_range:
                    # Parse range like "1:10", "5:", ":15"
                    if ":" in col_range:
                        parts = col_range.split(":")
                        start_idx = int(parts[0]) - 1 if parts[0] else 0  # Convert to 0-based
                        end_idx = int(parts[1]) if parts[1] else len(all_columns)

                        # Filter out _row_num_ from the range selection, we'll add it back later
                        columns_for_range = [col for col in all_columns if col != "_row_num_"]
                        selected_columns = columns_for_range[start_idx:end_idx]

                        # Always include _row_num_ at the beginning if it exists
                        if has_row_num:
                            columns_list = ["_row_num_"] + selected_columns
                        else:
                            columns_list = selected_columns
                    else:
                        console.print(
                            "[yellow]Warning: Invalid range format. Use 'start:end' format[/yellow]"
                        )
                elif col_first:
                    # Filter out _row_num_ from the first N selection, we'll add it back later
                    columns_for_first = [col for col in all_columns if col != "_row_num_"]
                    selected_columns = columns_for_first[:col_first]

                    # Always include _row_num_ at the beginning if it exists
                    if has_row_num:
                        columns_list = ["_row_num_"] + selected_columns
                    else:
                        columns_list = selected_columns
                elif col_last:
                    # Filter out _row_num_ from the last N selection, we'll add it back later
                    columns_for_last = [col for col in all_columns if col != "_row_num_"]
                    selected_columns = columns_for_last[-col_last:]

                    # Always include _row_num_ at the beginning if it exists
                    if has_row_num:
                        columns_list = ["_row_num_"] + selected_columns
                    else:
                        columns_list = selected_columns

        # Generate preview
        with console.status("[bold green]Generating preview..."):
            # Get total dataset size before preview and gather metadata
            try:
                # Process the data to get the actual data object for row count and metadata
                from pointblank.validate import (
                    _process_connection_string,
                    _process_csv_input,
                    _process_parquet_input,
                )

                processed_data = data
                if isinstance(data, str):
                    processed_data = _process_connection_string(data)
                    processed_data = _process_csv_input(processed_data)
                    processed_data = _process_parquet_input(processed_data)

                total_dataset_rows = pb.get_row_count(processed_data)

                # Determine source type and table type for enhanced preview title
                if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                    source_type = f"Pointblank dataset: {data_source}"
                else:
                    source_type = f"External source: {data_source}"

                table_type = _get_tbl_type(processed_data)
            except Exception:
                # If we can't get metadata, set defaults
                total_dataset_rows = None
                source_type = f"Data source: {data_source}"
                table_type = "unknown"

            gt_table = pb.preview(
                data=data,
                columns_subset=columns_list,
                n_head=head,
                n_tail=tail,
                limit=limit,
                show_row_numbers=not no_row_numbers,
                max_col_width=max_col_width,
                min_tbl_width=min_table_width,
                incl_header=not no_header,
            )

        if output_html:
            # Save HTML to file
            html_content = gt_table.as_raw_html()
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] HTML saved to: {output_html}")
        else:
            # Display in terminal with preview context info
            preview_info = None
            if total_dataset_rows is not None:
                # Determine if we're showing the complete dataset
                expected_rows = min(head + tail, limit, total_dataset_rows)
                is_complete = total_dataset_rows <= expected_rows

                preview_info = {
                    "total_rows": total_dataset_rows,
                    "head_rows": head,
                    "tail_rows": tail,
                    "is_complete": is_complete,
                    "source_type": source_type,
                    "table_type": table_type,
                }

            _rich_print_gt_table(gt_table, preview_info)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("data_source", type=str)
def info(data_source: str):
    """
    Display information about a data source.

    Shows table type, dimensions, column names, and data types.
    """
    try:
        with console.status("[bold green]Loading data..."):
            # Try to load as a pointblank dataset first
            if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                data = pb.load_dataset(data_source)
                source_type = f"Pointblank dataset: {data_source}"
            else:
                # Assume it's a file path or connection string
                data = data_source
                source_type = f"External source: {data_source}"

                # Process the data to get actual table object for inspection
                from pointblank.validate import (
                    _process_connection_string,
                    _process_csv_input,
                    _process_parquet_input,
                )

                data = _process_connection_string(data)
                data = _process_csv_input(data)
                data = _process_parquet_input(data)

        # Get table information
        tbl_type = _get_tbl_type(data)
        row_count = pb.get_row_count(data)
        col_count = pb.get_column_count(data)

        # Import the box style for consistent styling with scan table
        from rich.box import SIMPLE_HEAD

        # Create info table with same styling as scan table
        info_table = Table(
            title="Data Source Information",
            show_header=True,
            header_style="bold magenta",
            box=SIMPLE_HEAD,
            title_style="bold cyan",
            title_justify="left",
        )
        info_table.add_column("Property", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="green")

        info_table.add_row("Source", source_type)
        info_table.add_row("Table Type", tbl_type)
        info_table.add_row("Rows", f"{row_count:,}")
        info_table.add_row("Columns", f"{col_count:,}")

        console.print()
        console.print(info_table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("data_source", type=str)
@click.option("--output-html", type=click.Path(), help="Save HTML scan report to file")
@click.option("--columns", "-c", help="Comma-separated list of columns to scan")
def scan(
    data_source: str,
    output_html: str | None,
    columns: str | None,
):
    """
    Generate a data scan profile report.

    Produces a comprehensive data profile including:

    \b
    - Column types and distributions
    - Missing value patterns
    - Basic statistics
    - Data quality indicators

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
    """
    try:
        import time

        start_time = time.time()

        with console.status("[bold green]Loading data..."):
            # Try to load as a pointblank dataset first
            if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                data = pb.load_dataset(data_source)
                console.print(f"[green]✓[/green] Loaded dataset: {data_source}")
            else:
                # Assume it's a file path or connection string
                data = data_source
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Parse columns if provided
        columns_list = None
        if columns:
            columns_list = [col.strip() for col in columns.split(",")]

        # Generate data scan
        with console.status("[bold green]Generating data scan..."):
            # Use col_summary_tbl for comprehensive column scanning
            if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                # For pointblank datasets, data is already the loaded dataframe
                scan_result = pb.col_summary_tbl(data=data)
                source_type = f"Pointblank dataset: {data_source}"
                table_type = _get_tbl_type(data)
                # Get row count for footer
                try:
                    total_rows = pb.get_row_count(data)
                except Exception:
                    total_rows = None
            else:
                # For file paths and connection strings, load the data first
                from pointblank.validate import (
                    _process_connection_string,
                    _process_csv_input,
                    _process_parquet_input,
                )

                processed_data = _process_connection_string(data)
                processed_data = _process_csv_input(processed_data)
                processed_data = _process_parquet_input(processed_data)
                scan_result = pb.col_summary_tbl(data=processed_data)
                source_type = f"External source: {data_source}"
                table_type = _get_tbl_type(processed_data)
                # Get row count for footer
                try:
                    total_rows = pb.get_row_count(processed_data)
                except Exception:
                    total_rows = None

        scan_time = time.time() - start_time

        if output_html:
            # Save HTML to file
            try:
                html_content = scan_result.as_raw_html()
                Path(output_html).write_text(html_content, encoding="utf-8")
                console.print(f"[green]✓[/green] Data scan report saved to: {output_html}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save HTML report: {e}[/yellow]")
        else:
            # Display rich scan table in terminal
            console.print(f"[green]✓[/green] Data scan completed in {scan_time:.2f}s")
            console.print("Use --output-html to save the full interactive scan report.")

            # Display detailed column summary using rich formatting
            try:
                _rich_print_scan_table(
                    scan_result, data_source, source_type, table_type, total_rows
                )

            except Exception as e:
                console.print(f"[yellow]Could not display scan summary: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("data_source", type=str)
@click.option("--output-html", type=click.Path(), help="Save HTML output to file")
def missing(data_source: str, output_html: str | None):
    """
    Generate a missing values report for a data table.

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
    """
    try:
        with console.status("[bold green]Loading data..."):
            # Try to load as a pointblank dataset first
            if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                data = pb.load_dataset(data_source)
                console.print(f"[green]✓[/green] Loaded dataset: {data_source}")
            else:
                # Assume it's a file path or connection string
                data = data_source
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Generate missing values table
        with console.status("[bold green]Analyzing missing values..."):
            gt_table = pb.missing_vals_tbl(data)

            # Get original data for column types
            original_data = data
            if isinstance(data, str):
                # Process the data to get the actual data object
                from pointblank.validate import (
                    _process_connection_string,
                    _process_csv_input,
                    _process_parquet_input,
                )

                try:
                    original_data = _process_connection_string(data)
                    original_data = _process_csv_input(original_data)
                    original_data = _process_parquet_input(original_data)
                except Exception:
                    pass  # Use the string data as fallback

        if output_html:
            # Save HTML to file
            html_content = gt_table.as_raw_html()
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] Missing values report saved to: {output_html}")
        else:
            # Display in terminal with special missing values formatting
            _rich_print_missing_table(gt_table, original_data)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("output_file", type=click.Path())
def validate_example(output_file: str):
    """
    Generate an example validation script.

    Creates a sample Python script showing how to use Pointblank for validation.
    """
    example_script = '''"""
Example Pointblank validation script.

This script demonstrates how to create validation rules for your data.
Modify the validation rules below to match your data requirements.
"""

import pointblank as pb

# Create a validation object
# The 'data' variable is automatically provided by the CLI
validation = (
    pb.Validate(
        data=data,
        tbl_name="Example Data",
        label="CLI Validation Example",
        thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
    )
    # Add your validation rules here
    # Example rules (modify these based on your data structure):

    # Check that specific columns exist
    # .col_exists(["column1", "column2"])

    # Check for null values
    # .col_vals_not_null(columns="important_column")

    # Check value ranges
    # .col_vals_gt(columns="amount", value=0)
    # .col_vals_between(columns="score", left=0, right=100)

    # Check string patterns
    # .col_vals_regex(columns="email", pattern=r"^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$")

    # Check unique values
    # .col_vals_unique(columns="id")

    # Finalize the validation
    .interrogate()
)

# The validation object will be automatically used by the CLI
'''

    Path(output_file).write_text(example_script)
    console.print(f"[green]✓[/green] Example validation script created: {output_file}")
    console.print("\nEdit the script to add your validation rules, then run:")
    console.print(f"[cyan]pb validate your_data.csv {output_file}[/cyan]")


@cli.command()
@click.argument("data_source", type=str)
@click.argument("validation_script", type=click.Path(exists=True))
@click.option("--output-html", type=click.Path(), help="Save HTML validation report to file")
@click.option("--output-json", type=click.Path(), help="Save JSON validation summary to file")
@click.option("--fail-on-error", is_flag=True, help="Exit with non-zero code if validation fails")
def validate(
    data_source: str,
    validation_script: str,
    output_html: str | None,
    output_json: str | None,
    fail_on_error: bool,
):
    """
    Run validation using a Python validation script.

    DATA_SOURCE can be:

    \b
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

    VALIDATION_SCRIPT should be a Python file that defines validation rules.
    See 'pb validate-example' for a sample script.
    """
    try:
        with console.status("[bold green]Loading data..."):
            # Try to load as a pointblank dataset first
            if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                data = pb.load_dataset(data_source)
                console.print(f"[green]✓[/green] Loaded dataset: {data_source}")
            else:
                # Assume it's a file path or connection string
                data = data_source
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Execute the validation script
        with console.status("[bold green]Running validation..."):
            # Read and execute the validation script
            script_content = Path(validation_script).read_text()

            # Create a namespace with pointblank and the data
            namespace = {
                "pb": pb,
                "pointblank": pb,
                "data": data,
                "__name__": "__main__",
            }

            # Execute the script
            try:
                exec(script_content, namespace)
            except Exception as e:
                console.print(f"[red]Error executing validation script:[/red] {e}")
                sys.exit(1)

            # Look for a validation object in the namespace
            validation = None

            # Try to find the 'validation' variable specifically first
            if "validation" in namespace:
                validation = namespace["validation"]
            else:
                # Look for any validation object in the namespace
                for key, value in namespace.items():
                    if hasattr(value, "interrogate") and hasattr(value, "validation_info"):
                        validation = value
                        break
                    # Also check if it's a Validate object that has been interrogated
                    elif str(type(value)).find("Validate") != -1:
                        validation = value
                        break

            if validation is None:
                raise ValueError(
                    "No validation object found in script. "
                    "Script should create a Validate object and assign it to a variable named 'validation'."
                )

        console.print("[green]✓[/green] Validation completed")

        # Display summary
        _display_validation_summary(validation)

        # Save outputs
        if output_html:
            try:
                # Get HTML representation
                html_content = validation._repr_html_()
                Path(output_html).write_text(html_content, encoding="utf-8")
                console.print(f"[green]✓[/green] HTML report saved to: {output_html}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save HTML report: {e}[/yellow]")

        if output_json:
            try:
                # Get JSON report
                json_report = validation.get_json_report()
                Path(output_json).write_text(json_report, encoding="utf-8")
                console.print(f"[green]✓[/green] JSON summary saved to: {output_json}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save JSON report: {e}[/yellow]")

        # Check if we should fail on error
        if fail_on_error:
            try:
                if (
                    hasattr(validation, "validation_info")
                    and validation.validation_info is not None
                ):
                    info = validation.validation_info
                    n_critical = sum(1 for step in info if step.critical)
                    n_error = sum(1 for step in info if step.error)

                    if n_critical > 0 or n_error > 0:
                        severity = "critical" if n_critical > 0 else "error"
                        console.print(
                            f"[red]Exiting with error due to {severity} validation failures[/red]"
                        )
                        sys.exit(1)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not check validation status for fail-on-error: {e}[/yellow]"
                )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("data_source", type=str)
@click.argument("validation_script", type=click.Path(exists=True))
@click.argument("step_number", type=int)
@click.option(
    "--limit", "-l", default=100, help="Maximum number of failing rows to show (default: 100)"
)
@click.option("--output-csv", type=click.Path(), help="Save failing rows to CSV file")
@click.option("--output-html", type=click.Path(), help="Save failing rows table to HTML file")
def extract(
    data_source: str,
    validation_script: str,
    step_number: int,
    limit: int,
    output_csv: str | None,
    output_html: str | None,
):
    """
    Extract failing rows from a specific validation step.

    This command runs a validation and extracts the rows that failed
    a specific validation step, which is useful for debugging data quality issues.

    DATA_SOURCE: Same as validate command
    VALIDATION_SCRIPT: Path to validation script
    STEP_NUMBER: The step number to extract failing rows from (1-based)
    """
    try:
        with console.status("[bold green]Loading data..."):
            # Try to load as a pointblank dataset first
            if data_source in ["small_table", "game_revenue", "nycflights", "global_sales"]:
                data = pb.load_dataset(data_source)
                console.print(f"[green]✓[/green] Loaded dataset: {data_source}")
            else:
                # Assume it's a file path or connection string
                data = data_source
                console.print(f"[green]✓[/green] Loaded data source: {data_source}")

        # Execute the validation script
        with console.status("[bold green]Running validation..."):
            # Read and execute the validation script
            script_content = Path(validation_script).read_text()

            # Create a namespace with pointblank and the data
            namespace = {
                "pb": pb,
                "pointblank": pb,
                "data": data,
                "__name__": "__main__",
            }

            # Execute the script
            try:
                exec(script_content, namespace)
            except Exception as e:
                console.print(f"[red]Error executing validation script:[/red] {e}")
                sys.exit(1)

            # Look for a validation object in the namespace
            validation = None
            if "validation" in namespace:
                validation = namespace["validation"]
            else:
                # Look for any validation object in the namespace
                for key, value in namespace.items():
                    if hasattr(value, "interrogate") and hasattr(value, "validation_info"):
                        validation = value
                        break
                    elif str(type(value)).find("Validate") != -1:
                        validation = value
                        break

            if validation is None:
                raise ValueError(
                    "No validation object found in script. "
                    "Script should create a Validate object and assign it to a variable named 'validation'."
                )

        console.print("[green]✓[/green] Validation completed")

        # Extract failing rows from the specified step
        with console.status(f"[bold green]Extracting failing rows from step {step_number}..."):
            try:
                # Get the data extracts for the specific step
                step_extract = validation.get_data_extracts(i=step_number, frame=True)

                if step_extract is None or len(step_extract) == 0:
                    console.print(f"[yellow]No failing rows found for step {step_number}[/yellow]")
                    return

                # Limit the results
                if len(step_extract) > limit:
                    step_extract = step_extract.head(limit)
                    console.print(f"[yellow]Limited to first {limit} failing rows[/yellow]")

                console.print(f"[green]✓[/green] Extracted {len(step_extract)} failing rows")

                # Save outputs
                if output_csv:
                    if hasattr(step_extract, "write_csv"):
                        step_extract.write_csv(output_csv)
                    else:
                        step_extract.to_csv(output_csv, index=False)
                    console.print(f"[green]✓[/green] Failing rows saved to CSV: {output_csv}")

                if output_html:
                    # Create a preview of the failing rows
                    preview_table = pb.preview(
                        step_extract, n_head=min(10, len(step_extract)), n_tail=0
                    )
                    html_content = preview_table._repr_html_()
                    Path(output_html).write_text(html_content, encoding="utf-8")
                    console.print(
                        f"[green]✓[/green] Failing rows table saved to HTML: {output_html}"
                    )

                if not output_csv and not output_html:
                    # Display basic info about the failing rows
                    info_table = Table(
                        title=f"Failing Rows - Step {step_number}",
                        show_header=True,
                        header_style="bold red",
                    )
                    info_table.add_column("Property", style="cyan")
                    info_table.add_column("Value", style="white")

                    info_table.add_row("Total Failing Rows", f"{len(step_extract):,}")
                    info_table.add_row(
                        "Columns",
                        f"{len(step_extract.columns) if hasattr(step_extract, 'columns') else 'N/A'}",
                    )

                    console.print(info_table)
                    console.print(
                        "\n[dim]Use --output-csv or --output-html to save the failing rows.[/dim]"
                    )

            except Exception as e:
                console.print(f"[red]Error extracting failing rows:[/red] {e}")
                # Try to provide helpful information
                if hasattr(validation, "validation_info") and validation.validation_info:
                    max_step = len(validation.validation_info)
                    console.print(f"[yellow]Available steps: 1 to {max_step}[/yellow]")

                    # Show step information
                    steps_table = Table(title="Available Validation Steps", show_header=True)
                    steps_table.add_column("Step", style="cyan")
                    steps_table.add_column("Type", style="white")
                    steps_table.add_column("Column", style="green")
                    steps_table.add_column("Has Failures", style="yellow")

                    for i, step in enumerate(validation.validation_info, 1):
                        has_failures = "Yes" if not step.all_passed else "No"
                        steps_table.add_row(
                            str(i),
                            step.assertion_type,
                            str(step.column) if step.column else "—",
                            has_failures,
                        )

                    console.print(steps_table)
                sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def _format_missing_percentage(value: float) -> str:
    """Format missing value percentages for display.

    Args:
        value: The percentage value (0-100)

    Returns:
        Formatted string with proper percentage display
    """
    if value == 0.0:
        return "[green]●[/green]"  # Large green circle for no missing values
    elif value == 100.0:
        return "[red]●[/red]"  # Large red circle for completely missing values
    elif value < 1.0 and value > 0:
        return "<1%"  # Less than 1%
    elif value > 99.0 and value < 100.0:
        return ">99%"  # More than 99%
    else:
        return f"{int(round(value))}%"  # Round to nearest integer with % sign


def _rich_print_missing_table(gt_table: Any, original_data: Any = None) -> None:
    """Convert a missing values GT table to Rich table with special formatting.

    Args:
        gt_table: The GT table object for missing values
        original_data: The original data source to extract column types
    """
    try:
        # Extract the underlying data from the GT table
        df = None

        if hasattr(gt_table, "_tbl_data") and gt_table._tbl_data is not None:
            df = gt_table._tbl_data
        elif hasattr(gt_table, "_data") and gt_table._data is not None:
            df = gt_table._data
        elif hasattr(gt_table, "data") and gt_table.data is not None:
            df = gt_table.data

        if df is not None:
            # Create a Rich table with horizontal lines
            from rich.box import SIMPLE_HEAD

            rich_table = Table(show_header=True, header_style="bold magenta", box=SIMPLE_HEAD)

            # Get column names
            columns = []
            try:
                if hasattr(df, "columns"):
                    columns = list(df.columns)
                elif hasattr(df, "schema"):
                    columns = list(df.schema.names)
            except Exception as e:
                console.print(f"[red]Error getting columns:[/red] {e}")
                columns = []

            if not columns:
                columns = [f"Column {i + 1}" for i in range(10)]  # Fallback

            # Get original data to extract column types
            column_types = {}
            if original_data is not None:
                try:
                    # Get column types from original data
                    if hasattr(original_data, "columns"):
                        original_columns = list(original_data.columns)
                        column_types = _get_column_dtypes(original_data, original_columns)
                except Exception as e:
                    console.print(f"[red]Error getting column types:[/red] {e}")
                    pass  # Use empty dict as fallback

            # Add columns to Rich table with special formatting for missing values table
            sector_columns = [col for col in columns if col != "columns" and col.isdigit()]

            # Two separate columns: Column name (20 chars) and Data type (10 chars)
            rich_table.add_column("Column", style="cyan", no_wrap=True, width=20)
            rich_table.add_column("Type", style="yellow", no_wrap=True, width=10)

            # Sector columns: All same width, optimized for "100%" (4 chars + padding)
            for sector in sector_columns:
                rich_table.add_column(
                    sector,
                    style="cyan",
                    justify="center",
                    no_wrap=True,
                    width=5,  # Fixed width optimized for percentage values
                )

            # Convert data to rows with special formatting
            rows = []
            try:
                if hasattr(df, "to_dicts"):
                    data_dict = df.to_dicts()
                elif hasattr(df, "to_dict"):
                    data_dict = df.to_dict("records")
                else:
                    data_dict = []

                for i, row in enumerate(data_dict):
                    try:
                        # Each row should have: [column_name, data_type, sector1, sector2, ...]
                        column_name = str(row.get("columns", ""))

                        # Truncate column name to 20 characters with ellipsis if needed
                        if len(column_name) > 20:
                            truncated_name = column_name[:17] + "…"
                        else:
                            truncated_name = column_name

                        # Get data type for this column
                        if column_name in column_types:
                            dtype = column_types[column_name]
                            if len(dtype) > 10:
                                truncated_dtype = dtype[:9] + "…"
                            else:
                                truncated_dtype = dtype
                        else:
                            truncated_dtype = "?"

                        # Start building the row with column name and type
                        formatted_row = [truncated_name, truncated_dtype]

                        # Add sector values (formatted percentages)
                        for sector in sector_columns:
                            value = row.get(sector, 0.0)
                            if isinstance(value, (int, float)):
                                formatted_row.append(_format_missing_percentage(float(value)))
                            else:
                                formatted_row.append(str(value))

                        rows.append(formatted_row)

                    except Exception as e:
                        console.print(f"[red]Error processing row {i}:[/red] {e}")
                        continue

            except Exception as e:
                console.print(f"[red]Error extracting data:[/red] {e}")
                rows = [["Error extracting data", "?", *["" for _ in sector_columns]]]

            # Add rows to Rich table
            for row in rows:
                try:
                    rich_table.add_row(*row)
                except Exception as e:
                    console.print(f"[red]Error adding row:[/red] {e}")
                    break

            # Show the table with custom spanner header if we have sector columns
            if sector_columns:
                # Create a custom header line that shows the spanner
                header_parts = []
                header_parts.append(" " * 20)  # Space for Column header
                header_parts.append(" " * 10)  # Space for Type header

                # Left-align "Row Sectors" with the first numbered column
                row_sectors_text = "Row Sectors"
                header_parts.append(row_sectors_text)

                # Print the custom spanner header
                console.print("[dim]" + "  ".join(header_parts) + "[/dim]")

                # Add a horizontal rule below the spanner
                rule_parts = []
                rule_parts.append(" " * 20)  # Space for Column header
                rule_parts.append(" " * 10)  # Space for Type header

                # Use a fixed width horizontal rule for "Row Sectors"
                horizontal_rule = "─" * 20
                rule_parts.append(horizontal_rule)

                # Print the horizontal rule
                console.print("[dim]" + "  ".join(rule_parts) + "[/dim]")

            # Print the Rich table (will handle terminal width automatically)
            console.print(rich_table)
            footer_text = (
                "[dim]Symbols: [green]●[/green] = no missing values, "
                "[red]●[/red] = completely missing, "
                "<1% = less than 1% missing, "
                ">99% = more than 99% missing[/dim]"
            )
            console.print(footer_text)

        else:
            # Fallback to regular table display
            _rich_print_gt_table(gt_table)

    except Exception as e:
        console.print(f"[red]Error rendering missing values table:[/red] {e}")
        # Fallback to regular table display
        _rich_print_gt_table(gt_table)


def _rich_print_scan_table(
    scan_result: Any,
    data_source: str,
    source_type: str,
    table_type: str,
    total_rows: int | None = None,
) -> None:
    """
    Display scan results as a Rich table in the terminal with statistical measures.

    Args:
        scan_result: The GT object from col_summary_tbl()
        data_source: Name of the data source being scanned
        source_type: Type of data source (e.g., "Pointblank dataset: small_table")
        table_type: Type of table (e.g., "polars.LazyFrame")
        total_rows: Total number of rows in the dataset
    """
    try:
        import re

        import narwhals as nw
        from rich.box import SIMPLE_HEAD

        # Extract the underlying DataFrame from the GT object
        # The GT object has a _tbl_data attribute that contains the DataFrame
        gt_data = scan_result._tbl_data

        # Convert to Narwhals DataFrame for consistent handling
        nw_data = nw.from_native(gt_data)

        # Convert to dictionary for easier access
        data_dict = nw_data.to_dict(as_series=False)

        # Create main scan table with missing data table styling
        # Create a comprehensive title with data source, source type, and table type
        title_text = f"Column Summary / {source_type} / {table_type}"

        scan_table = Table(
            title=title_text,
            show_header=True,
            header_style="bold magenta",
            box=SIMPLE_HEAD,
            title_style="bold cyan",
            title_justify="left",
        )

        # Add columns with specific styling and appropriate widths
        scan_table.add_column("Column", style="cyan", no_wrap=True, width=20)
        scan_table.add_column("Type", style="yellow", no_wrap=True, width=10)
        scan_table.add_column(
            "NA", style="red", width=6, justify="right"
        )  # Adjusted for better formatting
        scan_table.add_column(
            "UQ", style="green", width=8, justify="right"
        )  # Adjusted for boolean values

        # Add statistical columns if they exist with appropriate widths
        stat_columns = []
        column_mapping = {
            "mean": ("Mean", "blue", 9),
            "std": ("SD", "blue", 9),
            "min": ("Min", "yellow", 9),
            "median": ("Med", "yellow", 9),
            "max": ("Max", "yellow", 9),
            "q_1": ("Q₁", "magenta", 8),
            "q_3": ("Q₃", "magenta", 9),
            "iqr": ("IQR", "magenta", 8),
        }

        for col_key, (display_name, color, width) in column_mapping.items():
            if col_key in data_dict:
                scan_table.add_column(display_name, style=color, width=width, justify="right")
                stat_columns.append(col_key)

        # Helper function to extract column name and type from HTML
        def extract_column_info(html_content: str) -> tuple[str, str]:
            """Extract column name and type from HTML formatted content."""
            # Extract column name from first div
            name_match = re.search(r"<div[^>]*>([^<]+)</div>", html_content)
            column_name = name_match.group(1) if name_match else "Unknown"

            # Extract data type from second div (with gray color)
            type_match = re.search(r"<div[^>]*color: gray[^>]*>([^<]+)</div>", html_content)
            if type_match:
                data_type = type_match.group(1)
                # Convert to compact format using the existing function
                compact_type = _format_dtype_compact(data_type)
                data_type = compact_type
            else:
                data_type = "unknown"

            return column_name, data_type

        # Helper function to format values with improved number formatting
        def format_value(
            value: Any, is_missing: bool = False, is_unique: bool = False, max_width: int = 8
        ) -> str:
            """Format values for display with smart number formatting and HTML cleanup."""
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return "[dim]—[/dim]"

            # Handle missing values indicator
            if is_missing and str(value) == "0":
                return "[green]●[/green]"  # No missing values

            # Clean up HTML formatting from the raw data
            str_val = str(value)

            # Handle multi-line values with <br> tags FIRST - take the first line (absolute number)
            if "<br>" in str_val:
                str_val = str_val.split("<br>")[0].strip()
                # For unique values, we want just the integer part
                if is_unique:
                    try:
                        # Try to extract just the integer part for unique counts
                        num_val = float(str_val)
                        return str(int(num_val))
                    except (ValueError, TypeError):
                        pass

            # Now handle HTML content (especially from boolean unique values)
            if "<" in str_val and ">" in str_val:
                # Remove HTML tags completely for cleaner display
                str_val = re.sub(r"<[^>]+>", "", str_val).strip()
                # Clean up extra whitespace
                str_val = re.sub(r"\s+", " ", str_val).strip()

            # Handle values like "2<.01" - extract the first number
            if "<" in str_val and not (str_val.startswith("<") and str_val.endswith(">")):
                # Extract number before the < symbol
                before_lt = str_val.split("<")[0].strip()
                if before_lt and before_lt.replace(".", "").replace("-", "").isdigit():
                    str_val = before_lt

            # Handle boolean unique values like "T0.62F0.38" - extract the more readable format
            if re.match(r"^[TF]\d+\.\d+[TF]\d+\.\d+$", str_val):
                # Extract T and F values
                t_match = re.search(r"T(\d+\.\d+)", str_val)
                f_match = re.search(r"F(\d+\.\d+)", str_val)
                if t_match and f_match:
                    t_val = float(t_match.group(1))
                    f_val = float(f_match.group(1))
                    # Show as "T0.62F0.38" but truncated if needed
                    formatted = f"T{t_val:.2f}F{f_val:.2f}"
                    if len(formatted) > max_width:
                        # Truncate to fit, showing dominant value
                        if t_val > f_val:
                            return f"T{t_val:.1f}"
                        else:
                            return f"F{f_val:.1f}"
                    return formatted

            # Try to parse as a number for better formatting
            try:
                # Try to convert to float first
                num_val = float(str_val)

                # Handle special cases
                if num_val == 0:
                    return "0"
                elif abs(num_val) == int(abs(num_val)) and abs(num_val) < 10000:
                    # Simple integers under 10000
                    return str(int(num_val))
                elif abs(num_val) >= 10000000 and abs(num_val) < 100000000:
                    # Likely dates in YYYYMMDD format - format as date-like
                    int_val = int(num_val)
                    if 19000101 <= int_val <= 29991231:  # Reasonable date range
                        str_date = str(int_val)
                        if len(str_date) == 8:
                            return (
                                f"{str_date[:4]}-{str_date[4:6]}-{str_date[6:]}"[: max_width - 1]
                                + "…"
                            )
                    # Otherwise treat as large number
                    return f"{num_val / 1000000:.1f}M"
                elif abs(num_val) >= 1000000:
                    # Large numbers - use scientific notation or M/k notation
                    if abs(num_val) >= 1000000000:
                        return f"{num_val:.1e}"
                    else:
                        return f"{num_val / 1000000:.1f}M"
                elif abs(num_val) >= 10000:
                    # Numbers >= 10k - use compact notation
                    return f"{num_val / 1000:.1f}k"
                elif abs(num_val) >= 100:
                    # Numbers 100-9999 - show with minimal decimals
                    return f"{num_val:.1f}"
                elif abs(num_val) >= 10:
                    # Numbers 10-99 - show with one decimal
                    return f"{num_val:.1f}"
                elif abs(num_val) >= 1:
                    # Numbers 1-9 - show with two decimals
                    return f"{num_val:.2f}"
                elif abs(num_val) >= 0.01:
                    # Small numbers - show with appropriate precision
                    return f"{num_val:.2f}"
                else:
                    # Very small numbers - use scientific notation
                    return f"{num_val:.1e}"

            except (ValueError, TypeError):
                # Not a number, handle as string
                pass

            # Handle date/datetime strings - show abbreviated format
            if len(str_val) > 10 and any(char in str_val for char in ["-", "/", ":"]):
                # Likely a date/datetime, show abbreviated
                if len(str_val) > max_width:
                    return str_val[: max_width - 1] + "…"

            # General string truncation with ellipsis
            if len(str_val) > max_width:
                return str_val[: max_width - 1] + "…"

            return str_val

        # Populate table rows
        num_rows = len(data_dict["colname"])
        for i in range(num_rows):
            row_data = []

            # Column name and type from HTML content
            colname_html = data_dict["colname"][i]
            column_name, data_type = extract_column_info(colname_html)
            row_data.append(column_name)
            row_data.append(data_type)

            # Missing values (NA)
            missing_val = data_dict.get("n_missing", [None] * num_rows)[i]
            row_data.append(format_value(missing_val, is_missing=True, max_width=6))

            # Unique values (UQ)
            unique_val = data_dict.get("n_unique", [None] * num_rows)[i]
            row_data.append(format_value(unique_val, is_unique=True, max_width=8))

            # Statistical columns
            for stat_col in stat_columns:
                stat_val = data_dict.get(stat_col, [None] * num_rows)[i]
                # Use appropriate width based on column type
                if stat_col in ["q_1", "iqr"]:
                    width = 8
                elif stat_col in ["mean", "std", "min", "median", "max", "q_3"]:
                    width = 9
                else:
                    width = 8
                row_data.append(format_value(stat_val, max_width=width))

            scan_table.add_row(*row_data)

        # Display the results
        console.print()
        console.print(scan_table)  # Add informational footer about the scan scope
        try:
            if total_rows is not None:
                # Full table scan
                footer_text = f"[dim]Scan from all {total_rows:,} rows in the table.[/dim]"

                # Create a simple footer
                footer_table = Table(
                    show_header=False,
                    show_lines=False,
                    box=None,
                    padding=(0, 0),
                )
                footer_table.add_column("", style="dim", width=80)
                footer_table.add_row(footer_text)
                console.print(footer_table)

        except Exception:
            # If we can't determine the scan scope, don't show a footer
            pass

    except Exception as e:
        # Fallback to simple message if table creation fails
        console.print(f"[yellow]Scan results available for {data_source}[/yellow]")
        console.print(f"[red]Error displaying table: {str(e)}[/red]")


if __name__ == "__main__":
    cli()
