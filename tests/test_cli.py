from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner
from rich.console import Console

import pointblank as pb
from pointblank.cli import (
    cli,
    datasets,
    requirements,
    preview,
    info,
    scan,
    missing,
    validate_example,
    validate,
    extract,
    validate_simple,
    _format_cell_value,
    _get_column_dtypes,
    _format_dtype_compact,
    _rich_print_gt_table,
    _display_validation_summary,
    _format_missing_percentage,
    _rich_print_missing_table,
    _rich_print_scan_table,
)
from pointblank._utils import _get_tbl_type, _is_lib_present


# ================================================================================
# UTILITY FUNCTION TESTS
# ================================================================================


def test_format_cell_value_basic():
    """Test basic cell value formatting."""
    # Test regular string
    assert _format_cell_value("test") == "test"

    # Test row number formatting
    assert _format_cell_value(123, is_row_number=True) == "[dim]123[/dim]"

    # Test None value
    assert _format_cell_value(None) == "[red]None[/red]"

    # Test empty string
    assert _format_cell_value("") == "[red][/red]"


def test_format_cell_value_truncation():
    """Test cell value truncation based on column count."""
    long_text = "a" * 100

    # Test with few columns (less aggressive truncation)
    result = _format_cell_value(long_text, max_width=50, num_columns=5)
    assert len(result) <= 50
    assert "…" in result

    # Test with many columns (more aggressive truncation)
    result = _format_cell_value(long_text, max_width=50, num_columns=20)
    assert len(result) <= 30
    assert "…" in result


@patch("pandas.isna")
@patch("numpy.isnan")
def test_format_cell_value_pandas_na(mock_isnan, mock_isna):
    """Test formatting of pandas/numpy NA values."""
    # Mock pandas NA detection
    mock_isna.return_value = True
    mock_isnan.return_value = True

    # Test NaN value
    result = _format_cell_value(float("nan"))
    # The function should detect NA values when pandas/numpy are available
    assert "[red]" in result


def test_format_dtype_compact():
    """Test data type formatting to compact representation."""
    # Test common type conversions
    assert _format_dtype_compact("utf8") == "str"
    assert _format_dtype_compact("string") == "str"
    assert _format_dtype_compact("int64") == "i64"
    assert _format_dtype_compact("int32") == "i32"
    assert _format_dtype_compact("float64") == "f64"
    assert _format_dtype_compact("float32") == "f32"
    assert _format_dtype_compact("boolean") == "bool"
    assert _format_dtype_compact("bool") == "bool"
    assert _format_dtype_compact("datetime") == "datetime"
    assert _format_dtype_compact("date") == "date"
    assert _format_dtype_compact("object") == "obj"
    assert _format_dtype_compact("category") == "cat"

    # Test unknown types with truncation for long names
    assert _format_dtype_compact("unknown_type") == "unknown_…"
    assert _format_dtype_compact("short") == "short"


def test_get_column_dtypes_pandas_like():
    """Test column dtype extraction for pandas-like objects."""
    # Simple test using actual pandas if available
    try:
        import pandas as pd

        df = pd.DataFrame({"col1": [1, 2], "col2": [1.0, 2.0], "col3": ["a", "b"]})
        columns = ["col1", "col2", "col3"]

        result = _get_column_dtypes(df, columns)

        # Should have entries for all columns
        assert len(result) == 3
        assert all(col in result for col in columns)
        assert all(result[col] != "?" for col in columns)  # Should detect types

    except ImportError:
        # Fallback test with mock
        mock_df = Mock()
        mock_df.dtypes = None
        columns = ["col1", "col2"]

        result = _get_column_dtypes(mock_df, columns)
        expected = {"col1": "?", "col2": "?"}
        assert result == expected


def test_get_column_dtypes_schema_based():
    """Test column dtype extraction for schema-based objects."""
    # Simplified test that exercises the fallback path
    mock_df = Mock()

    # Remove dtypes and schema to test fallback
    mock_df.dtypes = None
    mock_df.schema = None

    columns = ["col1", "col2", "col3"]

    result = _get_column_dtypes(mock_df, columns)
    expected = {"col1": "?", "col2": "?", "col3": "?"}
    assert result == expected


def test_get_column_dtypes_fallback():
    """Test fallback when no schema or dtypes available."""
    mock_df = Mock()
    mock_df.schema = None
    mock_df.dtypes = None

    columns = ["col1", "col2"]

    result = _get_column_dtypes(mock_df, columns)
    expected = {"col1": "?", "col2": "?"}
    assert result == expected


def test_format_missing_percentage():
    """Test missing percentage formatting."""
    assert _format_missing_percentage(0.0) == "[green]●[/green]"
    assert _format_missing_percentage(50.0) == "50%"
    assert _format_missing_percentage(33.3) == "33%"
    assert _format_missing_percentage(100.0) == "[red]●[/red]"
    assert _format_missing_percentage(0.5) == "<1%"
    assert _format_missing_percentage(99.5) == ">99%"


# ================================================================================
# RICH PRINT FUNCTION TESTS
# ================================================================================


@patch("pointblank.cli.console")
def test_rich_print_gt_table_basic(mock_console):
    """Test basic rich table printing."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table>data</table>"

    # Should not raise any exceptions
    _rich_print_gt_table(mock_gt_table)

    # Console should be used for printing
    mock_console.print.assert_called()


@patch("pointblank.cli.console")
def test_rich_print_gt_table_with_preview_info(mock_console):
    """Test rich table printing with preview information."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table>data</table>"

    preview_info = {"source_type": "Test Data", "shape": (100, 5), "table_type": "pandas.DataFrame"}

    _rich_print_gt_table(mock_gt_table, preview_info=preview_info)

    # Should print the table and info
    assert mock_console.print.call_count >= 2


@patch("pointblank.cli.console")
def test_rich_print_gt_table_error_handling(mock_console):
    """Test error handling in rich table printing."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.side_effect = Exception("HTML generation failed")

    # Should handle errors gracefully
    _rich_print_gt_table(mock_gt_table)

    # Should still attempt to print
    mock_console.print.assert_called()


@patch("pointblank.cli.console")
def test_display_validation_summary(mock_console):
    """Test validation summary display."""
    mock_validation = Mock()
    mock_validation.validation_info = [
        Mock(all_passed=True, n=100, n_passed=100, n_failed=0),
        Mock(all_passed=False, n=50, n_passed=45, n_failed=5),
    ]

    _display_validation_summary(mock_validation)

    # Should print summary information
    mock_console.print.assert_called()


@patch("pointblank.cli.console")
def test_display_validation_summary_no_info(mock_console):
    """Test validation summary display with no validation info."""
    mock_validation = Mock()
    mock_validation.validation_info = []

    _display_validation_summary(mock_validation)

    # Should handle empty validation info
    mock_console.print.assert_called()


@patch("pointblank.cli.console")
def test_rich_print_missing_table(mock_console):
    """Test missing values table printing."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table>missing data</table>"

    # Test with basic parameters (original_data is optional)
    _rich_print_missing_table(mock_gt_table)

    # Should print table
    mock_console.print.assert_called()


@patch("pointblank.cli.console")
def test_rich_print_scan_table(mock_console):
    """Test scan table printing."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table>scan data</table>"

    # Use correct parameter names
    _rich_print_scan_table(
        scan_result=mock_gt_table,
        data_source="test_data.csv",
        source_type="CSV",
        table_type="polars.DataFrame",
        total_rows=500,
    )

    # Should print table with footer information
    mock_console.print.assert_called()


# ================================================================================
# CLI COMMAND TESTS
# ================================================================================


def test_cli_main_help():
    """Test main CLI help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_version():
    """Test CLI version command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0


def test_datasets_command():
    """Test datasets listing command."""
    runner = CliRunner()
    result = runner.invoke(datasets)
    assert result.exit_code == 0
    assert "Available Pointblank Datasets" in result.output


@patch("pointblank.cli._is_lib_present")
def test_requirements_command(mock_is_lib_present):
    """Test requirements command."""
    # Mock library availability
    mock_is_lib_present.side_effect = lambda lib: lib in ["pandas", "polars"]

    runner = CliRunner()
    result = runner.invoke(requirements)
    assert result.exit_code == 0
    assert "Dependency Status" in result.output


def test_validate_example_command():
    """Test validate example command."""
    runner = CliRunner()
    result = runner.invoke(validate_example)
    # Exit code 2 is expected for Click parameter validation errors
    assert result.exit_code == 2


def test_extract_command():
    """Test extract command with basic functionality."""
    validation_script = """
# Simple validation script for testing
import pointblank as pb

validation = pb.Validate(data=pb.load_dataset("small_table"))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(validation_script)
        f.flush()

        runner = CliRunner()
        result = runner.invoke(extract, [f.name])
        # Exit code 2 is expected for Click parameter validation errors
        assert result.exit_code == 2


# ================================================================================
# ERROR HANDLING TESTS
# ================================================================================


def test_preview_command_exception():
    """Test preview command exception handling."""
    runner = CliRunner()

    # Test with non-existent dataset/file
    result = runner.invoke(preview, ["non_existent_file.csv"])
    assert result.exit_code == 1


def test_info_command_exception():
    """Test info command exception handling."""
    runner = CliRunner()

    # Test with non-existent dataset/file
    result = runner.invoke(info, ["non_existent_file.csv"])
    assert result.exit_code == 1


def test_scan_command_exception():
    """Test scan command exception handling."""
    runner = CliRunner()

    # Test with non-existent dataset/file
    result = runner.invoke(scan, ["non_existent_file.csv"])
    assert result.exit_code == 1


def test_missing_command_exception():
    """Test missing command exception handling."""
    runner = CliRunner()

    # Test with non-existent dataset/file
    result = runner.invoke(missing, ["non_existent_file.csv"])
    assert result.exit_code == 1


def test_validate_simple_exception():
    """Test validate_simple command exception handling."""
    runner = CliRunner()

    # Test with invalid parameters
    result = runner.invoke(validate_simple, ["non_existent_file.csv", "--check", "invalid-check"])
    # Exit code 2 is expected for Click parameter validation errors
    assert result.exit_code == 2


# ================================================================================
# EDGE CASE TESTS
# ================================================================================


def test_format_cell_value_pandas_na_edge_cases():
    """Test pandas NA handling edge cases."""
    # Test with actual pandas NA
    try:
        import pandas as pd
        import numpy as np

        # Test numpy nan
        result = _format_cell_value(np.nan)
        assert result == "[red]NaN[/red]"

        # Test pandas NA
        result = _format_cell_value(pd.NA)
        assert result == "[red]NA[/red]"

    except ImportError:
        # Skip if pandas/numpy not available
        pass


def test_format_cell_value_import_error_fallback():
    """Test fallback behavior in _format_cell_value."""
    # Test with normal string value
    result = _format_cell_value("normal_value")
    assert result == "normal_value"

    # Test with empty string
    result = _format_cell_value("")
    assert result == "[red][/red]"  # Empty string shown as red empty space

    # Test with numeric value
    result = _format_cell_value(123)
    assert result == "123"


def test_get_column_dtypes_pandas_edge_cases():
    """Test edge cases in pandas dtype detection."""
    # Create mock dataframe with edge cases
    mock_df = MagicMock()
    columns = ["col1", "col2", "col3"]

    # Test case where dtypes has fewer entries than columns
    mock_dtypes = MagicMock()
    mock_dtypes.__len__ = lambda: 2  # Fewer dtypes than columns

    # Create mock iloc with limited entries
    mock_iloc = MagicMock()
    mock_iloc.__len__ = lambda: 2
    mock_iloc.__getitem__ = lambda self, idx: MagicMock(__str__=lambda: f"dtype{idx}")
    mock_dtypes.iloc = mock_iloc

    mock_df.dtypes = mock_dtypes
    mock_df.schema = None  # To trigger pandas-like path

    result = _get_column_dtypes(mock_df, columns)
    # Should have "?" for columns beyond dtype length
    assert result["col3"] == "?"


def test_get_column_dtypes_no_iloc_fallback():
    """Test dtype access fallback when iloc is not available."""
    mock_df = MagicMock()
    columns = ["col1", "col2"]

    # Create dtypes object without iloc attribute
    mock_dtypes = MagicMock()
    mock_dtypes.__len__ = lambda: 2
    mock_dtypes.__getitem__ = lambda idx: MagicMock(__str__=lambda: f"dtype_{idx}")

    mock_df.dtypes = mock_dtypes
    mock_df.schema = None

    # Mock hasattr to return False for iloc
    with patch("pointblank.cli.hasattr", return_value=False):
        result = _get_column_dtypes(mock_df, columns)
        # Should use fallback access method
        assert len(result) == 2


# ================================================================================
# COMPLEX SCENARIO TESTS
# ================================================================================


def test_format_cell_value_edge_cases():
    """Test edge cases for cell value formatting."""
    # Test very long strings with different column counts
    very_long_text = "x" * 200

    # Few columns should allow more text
    result = _format_cell_value(very_long_text, num_columns=3)
    assert len(result) <= 50

    # Many columns should be more restrictive
    result = _format_cell_value(very_long_text, num_columns=25)
    assert len(result) <= 30

    # Test row numbers are never truncated
    long_row_number = 999999
    result = _format_cell_value(long_row_number, is_row_number=True)
    assert result == "[dim]999999[/dim]"


def test_rich_print_gt_table_wide_table():
    """Test rich table printing with wide tables."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table>" + "x" * 1000 + "</table>"

    # Should handle wide tables without issues
    with patch("pointblank.cli.console") as mock_console:
        _rich_print_gt_table(mock_gt_table)
        mock_console.print.assert_called()


def test_rich_print_scan_table_complex_data():
    """Test scan table with complex footer information."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table>complex scan</table>"

    with patch("pointblank.cli.console") as mock_console:
        _rich_print_scan_table(
            scan_result=mock_gt_table,
            data_source="complex_data.parquet",
            source_type="Complex Multi-table Join",
            table_type="polars.LazyFrame",
            total_rows=1000000,
        )
        mock_console.print.assert_called()
