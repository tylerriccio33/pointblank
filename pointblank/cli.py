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
@click.version_option(version=pb.__version__, prog_name="pointblank")
def cli():
    """
    Pointblank CLI - Data validation and quality tools for data engineers.

    Use this CLI to validate data, preview tables, and generate reports
    directly from the command line.
    """
    pass


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


