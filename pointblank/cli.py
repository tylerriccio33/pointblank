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
            # Create a Rich table
            rich_table = Table(show_header=True, header_style="bold magenta", box=None)

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

            for i, col in enumerate(display_columns):
                if col == "...more...":
                    # Add a special indicator column
                    rich_table.add_column("···", style="dim", width=3, no_wrap=True)
                else:
                    # Shorten the row number column name for better terminal display
                    display_col = "_i_" if col == "_row_num_" else str(col)
                    rich_table.add_column(
                        display_col, style="cyan", no_wrap=False, overflow="ellipsis"
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
                            [str(row.get(col, "")) for col in display_data_columns]
                            for row in data_dict
                        ]
                        # Add the "..." column in the middle
                        for i, row in enumerate(rows):
                            rows[i] = row[:7] + ["···"] + row[7:]
                    else:
                        rows = [[str(row.get(col, "")) for col in columns] for row in data_dict]
                elif hasattr(df, "to_dict"):
                    # Pandas-like interface
                    data_dict = df.to_dict("records")
                    if len(columns) > max_terminal_cols:
                        # For wide tables, extract only the displayed columns
                        display_data_columns = columns[:7] + columns[-7:]
                        rows = [
                            [str(row.get(col, "")) for col in display_data_columns]
                            for row in data_dict
                        ]
                        # Add the "..." column in the middle
                        for i, row in enumerate(rows):
                            rows[i] = row[:7] + ["···"] + row[7:]
                    else:
                        rows = [[str(row.get(col, "")) for col in columns] for row in data_dict]
                elif hasattr(df, "iter_rows"):
                    # Polars lazy frame
                    rows = [[str(val) for val in row] for row in df.iter_rows()]
                elif hasattr(df, "__iter__"):
                    # Try to iterate directly
                    rows = [[str(val) for val in row] for row in df]
                else:
                    rows = [["Could not extract data from this format"]]
            except Exception as e:
                rows = [[f"Error extracting data: {e}"]]

            # Add rows to Rich table (limit to prevent overwhelming output)
            max_rows = 50  # Reasonable limit for terminal display
            for i, row in enumerate(rows[:max_rows]):
                try:
                    rich_table.add_row(*row)
                except Exception as e:
                    # If there's an issue with row data, show error
                    rich_table.add_row(*[f"Error: {e}" for _ in columns])
                    break

            # Show the table
            console.print(rich_table)

            # Show summary info
            total_rows = len(rows)
            
            # Use preview info if available, otherwise fall back to old logic
            if preview_info:
                total_dataset_rows = preview_info.get('total_rows', total_rows)
                head_rows = preview_info.get('head_rows', 0)
                tail_rows = preview_info.get('tail_rows', 0)
                is_complete = preview_info.get('is_complete', False)
                
                if is_complete:
                    console.print(f"\n[dim]Showing all {total_rows} rows.[/dim]")
                elif head_rows > 0 and tail_rows > 0:
                    console.print(f"\n[dim]Showing first {head_rows} and last {tail_rows} rows from {total_dataset_rows:,} total rows.[/dim]")
                elif head_rows > 0:
                    console.print(f"\n[dim]Showing first {head_rows} rows from {total_dataset_rows:,} total rows.[/dim]")
                elif tail_rows > 0:
                    console.print(f"\n[dim]Showing last {tail_rows} rows from {total_dataset_rows:,} total rows.[/dim]")
                else:
                    # Fallback for other cases
                    console.print(f"\n[dim]Showing {total_rows} rows from {total_dataset_rows:,} total rows.[/dim]")
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
    - CSV file path (e.g., data.csv)
    - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
    - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
    - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

    COLUMN SELECTION OPTIONS:
    For tables with many columns, use these options to control which columns are displayed:

    --columns: Specify exact columns (e.g., --columns "name,age,email")
    --col-range: Select column range (e.g., --col-range "1:10", --col-range "5:", --col-range ":15")
    --col-first: Show first N columns (e.g., --col-first 5)
    --col-last: Show last N columns (e.g., --col-last 3)

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
                if col_range:
                    # Parse range like "1:10", "5:", ":15"
                    if ":" in col_range:
                        parts = col_range.split(":")
                        start_idx = int(parts[0]) - 1 if parts[0] else 0  # Convert to 0-based
                        end_idx = int(parts[1]) if parts[1] else len(all_columns)
                        columns_list = all_columns[start_idx:end_idx]
                    else:
                        console.print(
                            "[yellow]Warning: Invalid range format. Use 'start:end' format[/yellow]"
                        )
                elif col_first:
                    columns_list = all_columns[:col_first]
                elif col_last:
                    columns_list = all_columns[-col_last:]

        # Generate preview
        with console.status("[bold green]Generating preview..."):
            # Get total dataset size before preview
            try:
                # Process the data to get the actual data object for row count
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
            except Exception:
                # If we can't get row count, set to None
                total_dataset_rows = None
            
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
                    'total_rows': total_dataset_rows,
                    'head_rows': head,
                    'tail_rows': tail,
                    'is_complete': is_complete
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

        # Create info table
        info_table = Table(
            title="Data Source Information", show_header=True, header_style="bold magenta"
        )
        info_table.add_column("Property", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="green")

        info_table.add_row("Source", source_type)
        info_table.add_row("Table Type", tbl_type)
        info_table.add_row("Rows", f"{row_count:,}")
        info_table.add_row("Columns", f"{col_count:,}")

        console.print(info_table)

        # Show column information
        try:
            # Get column names
            if hasattr(data, "columns"):
                columns = list(data.columns)
            elif hasattr(data, "schema"):
                columns = list(data.schema.names)
            else:
                columns = ["Unable to determine columns"]

            if len(columns) <= 50:  # Only show if reasonable number of columns
                columns_table = Table(title="Columns", show_header=True, header_style="bold cyan")
                columns_table.add_column("Index", style="dim")
                columns_table.add_column("Column Name", style="green")
                columns_table.add_column("Data Type", style="yellow")

                # Try to get data types
                dtypes = []
                try:
                    if hasattr(data, "dtypes"):
                        # Polars/Pandas style
                        if hasattr(data.dtypes, "to_dict"):
                            dtypes_dict = data.dtypes.to_dict()
                            dtypes = [str(dtypes_dict.get(col, "Unknown")) for col in columns]
                        else:
                            dtypes = [str(dtype) for dtype in data.dtypes]
                    elif hasattr(data, "schema"):
                        # Other schema-based systems
                        schema = data.schema
                        if hasattr(schema, "to_dict"):
                            schema_dict = schema.to_dict()
                            dtypes = [str(schema_dict.get(col, "Unknown")) for col in columns]
                        else:
                            dtypes = [str(getattr(schema, col, "Unknown")) for col in columns]
                    else:
                        dtypes = ["Unknown"] * len(columns)
                except Exception:
                    dtypes = ["Unknown"] * len(columns)

                for idx, (col, dtype) in enumerate(zip(columns, dtypes)):
                    # Clean up dtype names for better display
                    dtype_clean = dtype.replace("polars.", "").replace("Utf8", "String")
                    columns_table.add_row(str(idx), col, dtype_clean)

                console.print(columns_table)
            else:
                console.print(
                    f"\n[yellow]Table has {len(columns)} columns (too many to display)[/yellow]"
                )
                console.print(f"First 10 columns: {', '.join(columns[:10])}")
                console.print(f"Last 10 columns: {', '.join(columns[-10:])}")

        except Exception as e:
            console.print(f"[yellow]Could not retrieve column information: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("data_source", type=str)
@click.option("--output-html", type=click.Path(), help="Save HTML scan report to file")
@click.option("--columns", "-c", help="Comma-separated list of columns to scan")
@click.option("--sample-size", default=10000, help="Sample size for scanning (default: 10000)")
def scan(
    data_source: str,
    output_html: str | None,
    columns: str | None,
    sample_size: int,
):
    """
    Generate a data scan profile report.

    Produces a comprehensive data profile including:
    - Column types and distributions
    - Missing value patterns
    - Basic statistics
    - Data quality indicators

    DATA_SOURCE can be:
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
            scan_result = pb.col_summary_tbl(data=data)
        
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
            # Display basic information in terminal
            console.print(f"[green]✓[/green] Data scan completed in {scan_time:.2f}s")
            console.print("Use --output-html to save the full interactive scan report.")

            # Show basic scan summary
            try:
                # Get basic info about the scan
                if hasattr(data, "shape"):
                    rows, cols = data.shape
                elif hasattr(data, "__len__") and hasattr(data, "columns"):
                    rows = len(data)
                    cols = len(data.columns)
                else:
                    rows = pb.get_row_count(data)
                    cols = pb.get_column_count(data)

                summary_table = Table(
                    title="Data Scan Summary", show_header=True, header_style="bold cyan"
                )
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="green")

                summary_table.add_row("Rows Scanned", f"{min(sample_size, rows):,}")
                summary_table.add_row("Total Rows", f"{rows:,}")
                summary_table.add_row(
                    "Columns Scanned", f"{len(columns_list) if columns_list else cols}"
                )
                summary_table.add_row("Sample Size", f"{sample_size:,}")

                console.print(summary_table)

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

        if output_html:
            # Save HTML to file
            html_content = gt_table.as_raw_html()
            Path(output_html).write_text(html_content, encoding="utf-8")
            console.print(f"[green]✓[/green] Missing values report saved to: {output_html}")
        else:
            # Display in terminal
            _rich_print_gt_table(gt_table)

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


if __name__ == "__main__":
    cli()
