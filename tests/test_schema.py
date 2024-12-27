import pathlib

import pytest

from pointblank.schema import Schema
from pointblank.validate import load_dataset

import pandas as pd
import polars as pl
import ibis


TBL_LIST = [
    "tbl_pd",
    "tbl_pl",
    "tbl_parquet",
    "tbl_duckdb",
    "tbl_sqlite",
]


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_parquet():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.parquet"
    return ibis.read_parquet(file_path)


@pytest.fixture
def tbl_duckdb():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.ddb"
    return ibis.connect(f"duckdb://{file_path}").table("tbl_xyz")


@pytest.fixture
def tbl_sqlite():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.sqlite"
    return ibis.sqlite.connect(file_path).table("tbl_xyz")


def test_schema_str(capfd):

    schema = Schema(columns=[("a", "int"), ("b", "str")])
    print(schema)
    captured = capfd.readouterr()
    expected_output = "Pointblank Schema\n  a: int\n  b: str\n"
    assert captured.out == expected_output


def test_schema_repr():
    schema = Schema(columns=[("a", "int"), ("b", "str")])
    expected_repr = "Schema(columns=[('a', 'int'), ('b', 'str')])"
    assert repr(schema) == expected_repr


def test_equivalent_inputs():
    schema_1 = Schema(columns=[("a", "int"), ("b", "str")])
    schema_2 = Schema(columns={"a": "int", "b": "str"})
    schema_3 = Schema(a="int", b="str")

    assert schema_1.columns == schema_2.columns
    assert schema_2.columns == schema_3.columns


def test_schema_from_pd_table():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))
    assert schema.columns == [
        ("date_time", "Datetime(time_unit='ns', time_zone=None)"),
        ("date", "Datetime(time_unit='ns', time_zone=None)"),
        ("a", "Int64"),
        ("b", "String"),
        ("c", "Float64"),
        ("d", "Float64"),
        ("e", "Boolean"),
        ("f", "String"),
    ]

    assert str(type(schema.tbl)) == "<class 'pandas.core.frame.DataFrame'>"


def test_schema_from_pl_table():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))
    assert schema.columns == [
        ("date_time", "Datetime(time_unit='us', time_zone=None)"),
        ("date", "Date"),
        ("a", "Int64"),
        ("b", "String"),
        ("c", "Int64"),
        ("d", "Float64"),
        ("e", "Boolean"),
        ("f", "String"),
    ]

    assert str(type(schema.tbl)) == "<class 'polars.dataframe.frame.DataFrame'>"


def test_schema_from_duckdb_table():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="duckdb"))
    assert schema.columns == [
        ("date_time", "timestamp(6)"),
        ("date", "date"),
        ("a", "int64"),
        ("b", "string"),
        ("c", "int64"),
        ("d", "float64"),
        ("e", "boolean"),
        ("f", "string"),
    ]

    assert str(type(schema.tbl)) == "<class 'ibis.expr.types.relations.Table'>"


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_schema_input_errors(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    with pytest.raises(ValueError):
        Schema()

    with pytest.raises(ValueError):
        Schema(tbl=tbl, columns=[("a", "int")])

    with pytest.raises(ValueError):
        Schema(columns=1)

    with pytest.raises(ValueError):
        Schema(tbl=tbl, a="int")
