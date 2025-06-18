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


def _rich_print_gt_table(gt_table: Any) -> None:
    """Convert a GT table to HTML and display it in the terminal."""
    try:
        # Get the HTML representation of the GT table
        html_content = gt_table.as_raw_html()

        # For now, we'll extract basic information and display as a Rich table
        # This is a simplified approach - in the future we could use a proper HTML renderer
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

        # Generate preview
        with console.status("[bold green]Generating preview..."):
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
            # Display in terminal
            _rich_print_gt_table(gt_table)

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

                for idx, col in enumerate(columns):
                    columns_table.add_row(str(idx), col)

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
            console.print("[green]✓[/green] Data scan completed")
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
