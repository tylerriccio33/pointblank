from __future__ import annotations
import pytest

from pointblank.compare import Compare
import polars.testing.parametric as pt
from hypothesis import given


@pytest.mark.xfail
def test_compare_basic(dfa, dfb) -> None:
    comp = Compare(dfa, dfb)

    comp.compare()

    raise NotImplementedError


def test_compare_csv_input():
    """Test Compare class with CSV file inputs."""
    # Test with CSV files
    csv_path1 = "data_raw/small_table.csv"
    csv_path2 = "data_raw/small_table.csv"  # Compare with itself for testing
    comp = Compare(csv_path1, csv_path2)
    comp.compare()  # Need to call compare() to create the scan objects
    assert comp._scana is not None
    assert comp._scanb is not None


def test_compare_parquet_input():
    """Test Compare class with Parquet file inputs."""
    # Test with Parquet files
    parquet_path1 = "tests/tbl_files/tbl_xyz.parquet"
    parquet_path2 = "tests/tbl_files/tbl_xyz.parquet"  # Compare with itself for testing
    comp = Compare(parquet_path1, parquet_path2)
    comp.compare()  # Need to call compare() to create the scan objects
    assert comp._scana is not None
    assert comp._scanb is not None


def test_compare_connection_string_input():
    """Test Compare class with connection string inputs."""
    # Test with connection strings
    conn1 = "duckdb:///Users/riannone/py_projects/pointblank/datasets/small_table.ddb::small_table"
    conn2 = "sqlite:///Users/riannone/py_projects/pointblank/tests/tbl_files/tbl_xyz.sqlite::tbl_xyz"
    comp = Compare(conn1, conn2)
    comp.compare()  # Need to call compare() to create the scan objects
    assert comp._scana is not None
    assert comp._scanb is not None


def test_compare_mixed_input_types():
    """Test Compare class with mixed input types."""
    # Test CSV vs Parquet
    csv_path = "data_raw/small_table.csv"
    parquet_path = "tests/tbl_files/tbl_xyz.parquet"
    comp = Compare(csv_path, parquet_path)
    comp.compare()  # Need to call compare() to create the scan objects
    assert comp._scana is not None
    assert comp._scanb is not None
    
    # Test connection string vs CSV
    conn = "duckdb:///Users/riannone/py_projects/pointblank/datasets/small_table.ddb::small_table"
    comp2 = Compare(conn, csv_path)
    comp2.compare()  # Need to call compare() to create the scan objects
    assert comp2._scana is not None
    assert comp2._scanb is not None
