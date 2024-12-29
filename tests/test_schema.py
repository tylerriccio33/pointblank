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


def test_schema_from_parquet_table(tbl_parquet):

    schema = Schema(tbl=tbl_parquet)

    assert schema.columns == [
        ("x", "int64"),
        ("y", "int64"),
        ("z", "int64"),
    ]

    assert str(type(schema.tbl)) == "<class 'ibis.expr.types.relations.Table'>"


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


def test_schema_from_sqlite_table(tbl_sqlite):

    schema = Schema(tbl=tbl_sqlite)

    assert schema.columns == [
        ("x", "int64"),
        ("y", "int64"),
        ("z", "int64"),
    ]

    assert str(type(schema.tbl)) == "<class 'ibis.expr.types.relations.Table'>"


def test_get_tbl_type_small_table():
    schema_pd = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))
    schema_pl = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))
    schema_duckdb = Schema(tbl=load_dataset(dataset="small_table", tbl_type="duckdb"))

    assert schema_pd.get_tbl_type() == "pandas"
    assert schema_pl.get_tbl_type() == "polars"
    assert schema_duckdb.get_tbl_type() == "duckdb"


def test_get_column_list_small_table():
    schema_pd = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))
    schema_pl = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))
    schema_duckdb = Schema(tbl=load_dataset(dataset="small_table", tbl_type="duckdb"))

    schemas = [schema_pd, schema_pl, schema_duckdb]

    for schema in schemas:
        assert schema.get_column_list() == [
            "date_time",
            "date",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        ]


def test_get_dtype_list_small_table_pd():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))

    assert schema.get_dtype_list() == [
        "Datetime(time_unit='ns', time_zone=None)",
        "Datetime(time_unit='ns', time_zone=None)",
        "Int64",
        "String",
        "Float64",
        "Float64",
        "Boolean",
        "String",
    ]


def test_get_dtype_list_small_table_pl():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))

    assert schema.get_dtype_list() == [
        "Datetime(time_unit='us', time_zone=None)",
        "Date",
        "Int64",
        "String",
        "Int64",
        "Float64",
        "Boolean",
        "String",
    ]


def test_get_dtype_list_small_table_duckdb():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="duckdb"))

    assert schema.get_dtype_list() == [
        "timestamp(6)",
        "date",
        "int64",
        "string",
        "int64",
        "float64",
        "boolean",
        "string",
    ]


def test_get_dtype_list_game_revenue_pd():
    schema = Schema(tbl=load_dataset(dataset="game_revenue", tbl_type="pandas"))

    assert schema.get_dtype_list() == [
        "String",
        "String",
        "Datetime(time_unit='ns', time_zone='UTC')",
        "Datetime(time_unit='ns', time_zone='UTC')",
        "String",
        "String",
        "Float64",
        "Float64",
        "Datetime(time_unit='ns', time_zone=None)",
        "String",
        "String",
    ]


def test_get_dtype_list_game_revenue_pl():
    schema = Schema(tbl=load_dataset(dataset="game_revenue", tbl_type="polars"))

    assert schema.get_dtype_list() == [
        "String",
        "String",
        "Datetime(time_unit='us', time_zone='UTC')",
        "Datetime(time_unit='us', time_zone='UTC')",
        "String",
        "String",
        "Float64",
        "Float64",
        "Date",
        "String",
        "String",
    ]


def test_get_dtype_list_game_revenue_duckdb():
    schema = Schema(tbl=load_dataset(dataset="game_revenue", tbl_type="duckdb"))

    assert schema.get_dtype_list() == [
        "string",
        "string",
        "timestamp('UTC', 6)",
        "timestamp('UTC', 6)",
        "string",
        "string",
        "float64",
        "float64",
        "date",
        "string",
        "string",
    ]


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

    with pytest.raises(ValueError):
        Schema(columns=("a", "int", "extra"))

    with pytest.raises(ValueError):
        Schema(columns=[("a", "int"), ["b", "str"]])

    with pytest.raises(ValueError):
        Schema(columns=("a", "int"))

    with pytest.raises(ValueError):
        Schema(columns=(1, "int"))
