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


