import pytest
import sys

from unittest.mock import patch

from great_tables import GT

from pointblank.validate import load_dataset
from pointblank.datascan import (
    DataScan,
    col_summary_tbl,
    _compact_0_1_fmt,
    _compact_decimal_fmt,
    _compact_integer_fmt,
)


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_class(tbl_type):
    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    scanner = DataScan(data=dataset)

    assert scanner.data.equals(dataset)
    assert scanner.tbl_name is None
    assert scanner.profile is not None
    assert isinstance(scanner.profile, dict)

    if tbl_type == "duckdb":
        assert scanner.tbl_type == "duckdb"
        assert scanner.tbl_category == "ibis"
        assert scanner.data_alt is None

    if tbl_type == "polars":
        assert scanner.tbl_type == "polars"
        assert scanner.tbl_category == "dataframe"
        assert scanner.data_alt is not None

    if tbl_type == "pandas":
        assert scanner.tbl_type == "pandas"
        assert scanner.tbl_category == "dataframe"
        assert scanner.data_alt is not None


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_class_use_tbl_name(tbl_type):
    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    scanner = DataScan(data=dataset, tbl_name="my_small_table")

    assert scanner.tbl_name == "my_small_table"


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_no_fail(tbl_type):
    small_table = load_dataset(dataset="small_table", tbl_type=tbl_type)
    DataScan(data=small_table)

    game_revenue = load_dataset(dataset="game_revenue", tbl_type=tbl_type)
    DataScan(data=game_revenue)

    nycflights = load_dataset(dataset="nycflights", tbl_type=tbl_type)
    DataScan(data=nycflights)


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_dict_output(tbl_type):
    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    scanner = DataScan(data=dataset)

    assert isinstance(scanner.to_dict(), dict)

    scan_dict = scanner.to_dict()

    assert isinstance(scan_dict, dict)

    assert scanner.to_dict() == scan_dict


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_json_output(tbl_type):
    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    scanner = DataScan(data=dataset)

    profile_json = scanner.to_json()

    assert isinstance(profile_json, str)


def test_datascan_json_file_output(tmp_path):
    dataset = load_dataset(dataset="small_table")
    scanner = DataScan(data=dataset)

    profile_json = scanner.to_json()

    file_path = tmp_path / "profile.json"
    scanner.save_to_json(output_file=file_path)

    assert file_path.exists()
    assert file_path.is_file()

    with open(file_path, "r") as f:
        file_content = f.read()

    assert profile_json == file_content


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_tabular_output_small_table(tbl_type):
    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    scanner = DataScan(data=dataset)

    tabular_output = scanner.get_tabular_report()

    assert isinstance(tabular_output, GT)


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_tabular_output_game_revenue(tbl_type):
    dataset = load_dataset(dataset="game_revenue", tbl_type=tbl_type)
    scanner = DataScan(data=dataset)

    tabular_output = scanner.get_tabular_report()

    assert isinstance(tabular_output, GT)


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_tabular_output_nycflights(tbl_type):
    dataset = load_dataset(dataset="nycflights", tbl_type=tbl_type)
    scanner = DataScan(data=dataset)

    tabular_output = scanner.get_tabular_report()

    assert isinstance(tabular_output, GT)


def test_col_summary_tbl():
    dataset = load_dataset(dataset="small_table")
    col_summary = col_summary_tbl(dataset)

    assert isinstance(col_summary, GT)


def test_col_summary_tbl_polars_categorical_column():
    import polars as pl

    log_levels = pl.Enum(["debug", "info", "warning", "error"])

    df_pl = pl.DataFrame(
        {
            "level": ["debug", "info", "debug", "error"],
            "message": [
                "process id: 525",
                "Service started correctly",
                "startup time: 67ms",
                "Cannot connect to DB!",
            ],
        },
        schema_overrides={
            "level": log_levels,
        },
    )

    tabular_output = col_summary_tbl(df_pl)

    assert isinstance(tabular_output, GT)


def test_col_summary_tbl_pandas_snap(snapshot):
    dataset = load_dataset(dataset="small_table", tbl_type="pandas")
    col_summary_html = col_summary_tbl(dataset).as_raw_html()

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(col_summary_html, "col_summary_html_pandas.html")


def test_col_summary_tbl_polars_snap(snapshot):
    dataset = load_dataset(dataset="small_table", tbl_type="polars")
    col_summary_html = col_summary_tbl(dataset).as_raw_html()

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(col_summary_html, "col_summary_html_polars.html")


def test_col_summary_tbl_duckdb_snap(snapshot):
    dataset = load_dataset(dataset="small_table", tbl_type="duckdb")
    col_summary_html = col_summary_tbl(dataset).as_raw_html()

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(col_summary_html, "col_summary_html_duckdb.html")


def test_datascan_class_raises():
    with pytest.raises(TypeError):
        DataScan(data="not a DataFrame or Ibis Table")

    with pytest.raises(TypeError):
        DataScan(data=123)

    with pytest.raises(TypeError):
        DataScan(data=[1, 2, 3])


def test_datascan_ibis_table_no_polars():
    # Mock the absence of the Polars library
    with patch.dict(sys.modules, {"polars": None}):
        small_table = load_dataset(dataset="small_table", tbl_type="duckdb")
        DataScan(data=small_table)


def test_compact_integer_fmt():
    assert _compact_integer_fmt(value=0) == "0"
    assert _compact_integer_fmt(value=0.4) == "4.0E−1"
    assert _compact_integer_fmt(value=0.6) == "6.0E−1"
    assert _compact_integer_fmt(value=1) == "1"
    assert _compact_integer_fmt(value=43.91) == "44"
    assert _compact_integer_fmt(value=226.1) == "226"
    assert _compact_integer_fmt(value=4362.54) == "4363"
    assert _compact_integer_fmt(value=15321.23) == "1.5E4"


def test_compact_decimal_fmt():
    assert _compact_decimal_fmt(value=0) == "0.00"
    assert _compact_decimal_fmt(value=1) == "1.00"
    assert _compact_decimal_fmt(value=0.0) == "0.00"
    assert _compact_decimal_fmt(value=1.0) == "1.00"
    assert _compact_decimal_fmt(value=0.1) == "0.10"
    assert _compact_decimal_fmt(value=0.5) == "0.50"
    assert _compact_decimal_fmt(value=0.01) == "0.01"
    assert _compact_decimal_fmt(value=0.009) == "9.0E−3"
    assert _compact_decimal_fmt(value=0.000001) == "1.0E−6"
    assert _compact_decimal_fmt(value=0.99) == "0.99"
    assert _compact_decimal_fmt(value=1) == "1.00"
    assert _compact_decimal_fmt(value=43.91) == "43.9"
    assert _compact_decimal_fmt(value=226.1) == "226"
    assert _compact_decimal_fmt(value=4362.54) == "4363"
    assert _compact_decimal_fmt(value=15321.23) == "1.5E4"


def test_compact_0_1_fmt():
    assert _compact_0_1_fmt(value=0) == " 0.00"
    assert _compact_0_1_fmt(value=1) == " 1.00"
    assert _compact_0_1_fmt(value=0.0) == " 0.00"
    assert _compact_0_1_fmt(value=1.0) == " 1.00"
    assert _compact_0_1_fmt(value=0.1) == " 0.10"
    assert _compact_0_1_fmt(value=0.5) == " 0.50"
    assert _compact_0_1_fmt(value=0.01) == " 0.01"
    assert _compact_0_1_fmt(value=0.009) == "<0.01"
    assert _compact_0_1_fmt(value=0.000001) == "<0.01"
    assert _compact_0_1_fmt(value=0.99) == " 0.99"
    assert _compact_0_1_fmt(value=0.995) == ">0.99"
    assert _compact_0_1_fmt(value=226.1) == "226"
