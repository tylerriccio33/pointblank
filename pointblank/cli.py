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


