import pathlib

import sys
from unittest.mock import patch
import pytest

import pandas as pd
import polars as pl
import ibis

import great_tables as GT

from pointblank.validate import Validate, _ValidationInfo, load_dataset
from pointblank.thresholds import Thresholds


TBL_LIST = [
    "tbl_pd",
    "tbl_pl",
    "tbl_parquet",
    "tbl_duckdb",
    "tbl_sqlite",
]

TBL_MISSING_LIST = [
    "tbl_missing_pd",
    "tbl_missing_pl",
    "tbl_missing_parquet",
    "tbl_missing_duckdb",
    "tbl_missing_sqlite",
]

TBL_DATES_TIMES_TEXT_LIST = [
    "tbl_dates_times_text_pd",
    "tbl_dates_times_text_pl",
    "tbl_dates_times_text_parquet",
    "tbl_dates_times_text_duckdb",
    "tbl_dates_times_text_sqlite",
]


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pd():
    return pd.DataFrame({"x": [1, 2, pd.NA, 4], "y": [4, pd.NA, 6, 7], "z": [8, pd.NA, 8, 8]})


@pytest.fixture
def tbl_dates_times_text_pd():
    return pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-02-01", pd.NA],
            "dttm": ["2021-01-01 00:00:00", pd.NA, "2021-02-01 00:00:00"],
            "text": [pd.NA, "5-egh-163", "8-kdg-938"],
        }
    )


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pl():
    return pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, None, 8, 8]})


@pytest.fixture
def tbl_dates_times_text_pl():
    return pl.DataFrame(
        {
            "date": ["2021-01-01", "2021-02-01", None],
            "dttm": ["2021-01-01 00:00:00", None, "2021-02-01 00:00:00"],
            "text": [None, "5-egh-163", "8-kdg-938"],
        }
    )


@pytest.fixture
def tbl_parquet():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.parquet"
    return ibis.read_parquet(file_path)


@pytest.fixture
def tbl_missing_parquet():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz_missing.parquet"
    return ibis.read_parquet(file_path)


@pytest.fixture
def tbl_dates_times_text_parquet():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_dates_times_text.parquet"
    return ibis.read_parquet(file_path)


@pytest.fixture
def tbl_duckdb():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.ddb"
    return ibis.connect(f"duckdb://{file_path}").table("tbl_xyz")


@pytest.fixture
def tbl_missing_duckdb():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz_missing.ddb"
    return ibis.connect(f"duckdb://{file_path}").table("tbl_xyz_missing")


@pytest.fixture
def tbl_dates_times_text_duckdb():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_dates_times_text.ddb"
    return ibis.connect(f"duckdb://{file_path}").table("tbl_dates_times_text")


@pytest.fixture
def tbl_sqlite():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.sqlite"
    return ibis.sqlite.connect(file_path).table("tbl_xyz")


@pytest.fixture
def tbl_missing_sqlite():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz_missing.sqlite"
    return ibis.sqlite.connect(file_path).table("tbl_xyz_missing")


@pytest.fixture
def tbl_dates_times_text_sqlite():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_dates_times_text.sqlite"
    return ibis.sqlite.connect(file_path).table("tbl_dates_times_text")


def test_validation_info():

    v = _ValidationInfo(
        i=1,
        i_o=1,
        step_id="col_vals_gt",
        sha1="a",
        assertion_type="col_vals_gt",
        column="x",
        values=0,
        inclusive=True,
        na_pass=False,
        thresholds=Thresholds(),
        label=None,
        brief=None,
        active=True,
        all_passed=True,
        n=4,
        n_passed=4,
        n_failed=0,
        f_passed=1.0,
        f_failed=0.0,
        warn=None,
        stop=None,
        notify=None,
        time_processed="2021-08-01T00:00:00",
        proc_duration_s=0.0,
    )

    assert v.i == 1
    assert v.i_o == 1
    assert v.step_id == "col_vals_gt"
    assert v.sha1 == "a"
    assert v.assertion_type == "col_vals_gt"
    assert v.column == "x"
    assert v.values == 0
    assert v.inclusive is True
    assert v.na_pass is False
    assert v.thresholds == Thresholds()
    assert v.label is None
    assert v.brief is None
    assert v.active is True
    assert v.all_passed is True
    assert v.n == 4
    assert v.n_passed == 4
    assert v.n_failed == 0
    assert v.f_passed == 1.0
    assert v.f_failed == 0.0
    assert v.warn is None
    assert v.stop is None
    assert v.notify is None

    assert isinstance(v.time_processed, str)
    assert isinstance(v.proc_duration_s, float)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_col_vals_all_passing(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    if tbl_fixture not in ["tbl_parquet", "tbl_duckdb", "tbl_sqlite"]:
        assert v.data.shape == (4, 3)
        assert str(v.data["x"].dtype).lower() == "int64"
        assert str(v.data["y"].dtype).lower() == "int64"
        assert str(v.data["z"].dtype).lower() == "int64"

    # There is a single validation check entry in the `validation_info` attribute
    assert len(v.validation_info) == 1

    # The single step had no failing test units so the `all_passed` attribute is `True`
    assert v.all_passed()

    # Test other validation types for all passing behavior in single steps
    assert Validate(tbl).col_vals_lt(columns="x", value=5).interrogate().all_passed()
    assert Validate(tbl).col_vals_eq(columns="z", value=8).interrogate().all_passed()
    assert Validate(tbl).col_vals_ge(columns="x", value=1).interrogate().all_passed()
    assert Validate(tbl).col_vals_le(columns="x", value=4).interrogate().all_passed()
    assert Validate(tbl).col_vals_between(columns="x", left=0, right=5).interrogate().all_passed()
    assert Validate(tbl).col_vals_outside(columns="x", left=-5, right=0).interrogate().all_passed()
    assert (
        Validate(tbl).col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5]).interrogate().all_passed()
    )
    assert Validate(tbl).col_vals_not_in_set(columns="x", set=[5, 6, 7]).interrogate().all_passed()


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_plan(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Create a validation plan
    v = Validate(tbl).col_vals_gt(columns="x", value=0)

    # A single validation step was added to the plan so `validation_info` has a single entry
    assert len(v.validation_info) == 1

    # Extract the `validation_info` object to check its attributes
    val_info = v.validation_info[0]

    assert [
        attr
        for attr in val_info.__dict__.keys()
        if not attr.startswith("__") and not attr.endswith("__")
    ] == [
        "i",
        "i_o",
        "step_id",
        "sha1",
        "assertion_type",
        "column",
        "values",
        "inclusive",
        "na_pass",
        "pre",
        "thresholds",
        "label",
        "brief",
        "active",
        "all_passed",
        "n",
        "n_passed",
        "n_failed",
        "f_passed",
        "f_failed",
        "warn",
        "stop",
        "notify",
        "tbl_checked",
        "extract",
        "time_processed",
        "proc_duration_s",
    ]

    # Check the attributes of the `validation_info` object
    assert val_info.i == 1
    assert val_info.assertion_type == "col_vals_gt"
    assert val_info.column == "x"
    assert val_info.values == 0
    assert val_info.na_pass is False
    assert val_info.thresholds == Thresholds()
    assert val_info.label is None
    assert val_info.brief is None
    assert val_info.active is True
    assert val_info.all_passed is None
    assert val_info.n is None
    assert val_info.n_passed is None
    assert val_info.n_failed is None
    assert val_info.f_passed is None
    assert val_info.f_failed is None
    assert val_info.warn is None
    assert val_info.stop is None
    assert val_info.notify is None
    assert val_info.tbl_checked is None
    assert val_info.extract is None
    assert val_info.time_processed is None
    assert val_info.proc_duration_s is None

    # Interrogate the validation plan
    v_int = v.interrogate()

    # The length of the validation info list is still 1
    assert len(v_int.validation_info) == 1

    # Extract the validation info object to check its attributes
    val_info_int = v.validation_info[0]

    # The attribute names of `validation_info` object are the same as before
    assert [
        attr
        for attr in val_info_int.__dict__.keys()
        if not attr.startswith("__") and not attr.endswith("__")
    ] == [
        "i",
        "i_o",
        "step_id",
        "sha1",
        "assertion_type",
        "column",
        "values",
        "inclusive",
        "na_pass",
        "pre",
        "thresholds",
        "label",
        "brief",
        "active",
        "all_passed",
        "n",
        "n_passed",
        "n_failed",
        "f_passed",
        "f_failed",
        "warn",
        "stop",
        "notify",
        "tbl_checked",
        "extract",
        "time_processed",
        "proc_duration_s",
    ]

    # Check the attributes of the `validation_info` object
    assert val_info.i == 1
    assert val_info.assertion_type == "col_vals_gt"
    assert val_info.column == "x"
    assert val_info.values == 0
    assert val_info.na_pass is False
    assert val_info.thresholds == Thresholds()
    assert val_info.label is None
    assert val_info.brief is None
    assert val_info.active is True
    assert val_info.all_passed is True
    assert val_info.n == 4
    assert val_info.n_passed == 4
    assert val_info.n_failed == 0
    assert val_info.f_passed == 1.0
    assert val_info.f_failed == 0.0
    assert val_info.warn is None
    assert val_info.stop is None
    assert val_info.notify is None
    assert isinstance(val_info.time_processed, str)
    assert val_info.proc_duration_s > 0.0


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_attr_getters(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    # Get the total number of test units as a dictionary
    n_dict = v.n()
    assert len(n_dict) == 1
    assert n_dict.keys() == {1}
    assert n_dict[1] == 4

    # Get the number of passing test units
    n_passed_dict = v.n_passed()
    assert len(n_passed_dict) == 1
    assert n_passed_dict.keys() == {1}
    assert n_passed_dict[1] == 4

    # Get the number of failing test units
    n_failed_dict = v.n_failed()
    assert len(n_failed_dict) == 1
    assert n_failed_dict.keys() == {1}
    assert n_failed_dict[1] == 0

    # Get the fraction of passing test units
    f_passed_dict = v.f_passed()
    assert len(f_passed_dict) == 1
    assert f_passed_dict.keys() == {1}
    assert f_passed_dict[1] == 1.0

    # Get the fraction of failing test units
    f_failed_dict = v.f_failed()
    assert len(f_failed_dict) == 1
    assert f_failed_dict.keys() == {1}
    assert f_failed_dict[1] == 0.0

    # Get the warn status
    warn_dict = v.warn()
    assert len(warn_dict) == 1
    assert warn_dict.keys() == {1}
    assert warn_dict[1] is None

    # Get the stop status
    stop_dict = v.stop()
    assert len(stop_dict) == 1
    assert stop_dict.keys() == {1}
    assert stop_dict[1] is None

    # Get the notify status
    notify_dict = v.notify()
    assert len(notify_dict) == 1
    assert notify_dict.keys() == {1}
    assert notify_dict[1] is None


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_report(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    assert v.get_json_report() != v.get_json_report(
        exclude_fields=["time_processed", "proc_duration_s"]
    )

    # A ValueError is raised when `use_fields=` includes invalid fields
    with pytest.raises(ValueError):
        v.get_json_report(use_fields=["invalid_field"])

    # A ValueError is raised when `exclude_fields=` includes invalid fields
    with pytest.raises(ValueError):
        v.get_json_report(exclude_fields=["invalid_field"])

    # A ValueError is raised `use_fields=` and `exclude_fields=` are both provided
    with pytest.raises(ValueError):
        v.get_json_report(use_fields=["i"], exclude_fields=["i_o"])


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_report_interrogate_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(columns="x", value=0)
        .interrogate()
        .get_json_report(exclude_fields=["time_processed", "proc_duration_s"])
    )

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(report, "validation_report.json")


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_report_no_interrogate_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(columns="x", value=0)
        .get_json_report(exclude_fields=["time_processed", "proc_duration_s"])
    )

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(report, "validation_report.json")


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_report_use_fields_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(columns="x", value=0)
        .get_json_report(
            use_fields=[
                "i",
                "assertion_type",
                "all_passed",
                "n",
                "f_passed",
                "f_failed",
            ]
        )
    )

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(report, "validation_report.json")


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_column_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `column=` is not a string
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(columns=9, left=0, right=5)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(columns=9, left=-5, right=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_in_set(columns=9, set=[1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_not_in_set(columns=9, set=[5, 6, 7])


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_na_pass_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `na_pass=` is not a boolean
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(columns="x", left=0, right=5, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(columns="x", left=-5, right=0, na_pass=9)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_thresholds_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Check that allowed forms for `thresholds=` don't raise an error
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=1)
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=0.1)
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(0.1))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(0.1, 0.2))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(0.1, 0.2, 0.3))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(0.1, 2, 0.3))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, 2))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, 3, 4))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, 0.3, 4))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds={"warn_at": 0.1})
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds={"stop_at": 0.1})
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds={"notify_at": 0.1})
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds={"warn_at": 0.05, "notify_at": 0.1})
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=Thresholds())
    Validate(tbl).col_vals_gt(
        columns="x", value=0, thresholds=Thresholds(warn_at=0.1, stop_at=0.2, notify_at=0.3)
    )
    Validate(tbl).col_vals_gt(
        columns="x", value=0, thresholds=Thresholds(warn_at=1, stop_at=2, notify_at=3)
    )

    # Raise a ValueError when `thresholds=` is not one of the allowed types
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds="invalid")
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=[1, 2, 3])
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=-2)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=())
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, -2))
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, [2], 3))
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(
            columns="x", value=0, thresholds={"warning": 0.05, "notify_at": 0.1}
        )
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(
            columns="x", value=0, thresholds={"warn_at": 0.05, "notify_at": -0.1}
        )
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(
            columns="x", value=0, thresholds={"warn_at": "invalid", "stop_at": 3}
        )


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_active_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `active=` is not a boolean
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(columns="x", left=0, right=5, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(columns="x", left=-5, right=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5], active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_not_in_set(columns="x", set=[5, 6, 7], active=9)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_thresholds_inherit(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Check that the `thresholds=` argument is inherited from Validate, in those steps where
    # it is not explicitly provided (is `None`)
    v = (
        Validate(tbl, thresholds=Thresholds(warn_at=1, stop_at=2, notify_at=3))
        .col_vals_gt(columns="x", value=0)
        .col_vals_gt(columns="x", value=0, thresholds=0.5)
        .col_vals_lt(columns="x", value=2)
        .col_vals_lt(columns="x", value=2, thresholds=0.5)
        .col_vals_eq(columns="z", value=4)
        .col_vals_eq(columns="z", value=4, thresholds=0.5)
        .col_vals_ne(columns="z", value=6)
        .col_vals_ne(columns="z", value=6, thresholds=0.5)
        .col_vals_ge(columns="z", value=8)
        .col_vals_ge(columns="z", value=8, thresholds=0.5)
        .col_vals_le(columns="z", value=10)
        .col_vals_le(columns="z", value=10, thresholds=0.5)
        .col_vals_between(columns="x", left=0, right=5)
        .col_vals_between(columns="x", left=0, right=5, thresholds=0.5)
        .col_vals_outside(columns="x", left=-5, right=0)
        .col_vals_outside(columns="x", left=-5, right=0, thresholds=0.5)
        .col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5])
        .col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5], thresholds=0.5)
        .col_vals_not_in_set(columns="x", set=[5, 6, 7])
        .col_vals_not_in_set(columns="x", set=[5, 6, 7], thresholds=0.5)
        .col_vals_null(columns="x")
        .col_vals_null(columns="x", thresholds=0.5)
        .col_vals_not_null(columns="x")
        .col_vals_not_null(columns="x", thresholds=0.5)
        .col_exists(columns="x")
        .col_exists(columns="x", thresholds=0.5)
        .interrogate()
    )

    # col_vals_gt - inherited
    assert v.validation_info[0].thresholds.warn_at == 1
    assert v.validation_info[0].thresholds.stop_at == 2
    assert v.validation_info[0].thresholds.notify_at == 3

    # col_vals_gt - overridden
    assert v.validation_info[1].thresholds.warn_at == 0.5
    assert v.validation_info[1].thresholds.stop_at is None
    assert v.validation_info[1].thresholds.notify_at is None

    # col_vals_lt - inherited
    assert v.validation_info[2].thresholds.warn_at == 1
    assert v.validation_info[2].thresholds.stop_at == 2
    assert v.validation_info[2].thresholds.notify_at == 3

    # col_vals_lt - overridden
    assert v.validation_info[3].thresholds.warn_at == 0.5
    assert v.validation_info[3].thresholds.stop_at is None
    assert v.validation_info[3].thresholds.notify_at is None

    # col_vals_eq - inherited
    assert v.validation_info[4].thresholds.warn_at == 1
    assert v.validation_info[4].thresholds.stop_at == 2
    assert v.validation_info[4].thresholds.notify_at == 3

    # col_vals_eq - overridden
    assert v.validation_info[5].thresholds.warn_at == 0.5
    assert v.validation_info[5].thresholds.stop_at is None
    assert v.validation_info[5].thresholds.notify_at is None

    # col_vals_ne - inherited
    assert v.validation_info[6].thresholds.warn_at == 1
    assert v.validation_info[6].thresholds.stop_at == 2
    assert v.validation_info[6].thresholds.notify_at == 3

    # col_vals_ne - overridden
    assert v.validation_info[7].thresholds.warn_at == 0.5
    assert v.validation_info[7].thresholds.stop_at is None
    assert v.validation_info[7].thresholds.notify_at is None

    # col_vals_ge - inherited
    assert v.validation_info[8].thresholds.warn_at == 1
    assert v.validation_info[8].thresholds.stop_at == 2
    assert v.validation_info[8].thresholds.notify_at == 3

    # col_vals_ge - overridden
    assert v.validation_info[9].thresholds.warn_at == 0.5
    assert v.validation_info[9].thresholds.stop_at is None
    assert v.validation_info[9].thresholds.notify_at is None

    # col_vals_le - inherited
    assert v.validation_info[10].thresholds.warn_at == 1
    assert v.validation_info[10].thresholds.stop_at == 2
    assert v.validation_info[10].thresholds.notify_at == 3

    # col_vals_le - overridden
    assert v.validation_info[11].thresholds.warn_at == 0.5
    assert v.validation_info[11].thresholds.stop_at is None
    assert v.validation_info[11].thresholds.notify_at is None

    # col_vals_between - inherited
    assert v.validation_info[12].thresholds.warn_at == 1
    assert v.validation_info[12].thresholds.stop_at == 2
    assert v.validation_info[12].thresholds.notify_at == 3

    # col_vals_between - overridden
    assert v.validation_info[13].thresholds.warn_at == 0.5
    assert v.validation_info[13].thresholds.stop_at is None
    assert v.validation_info[13].thresholds.notify_at is None

    # col_vals_outside - inherited
    assert v.validation_info[14].thresholds.warn_at == 1
    assert v.validation_info[14].thresholds.stop_at == 2
    assert v.validation_info[14].thresholds.notify_at == 3

    # col_vals_outside - overridden
    assert v.validation_info[15].thresholds.warn_at == 0.5
    assert v.validation_info[15].thresholds.stop_at is None
    assert v.validation_info[15].thresholds.notify_at is None

    # col_vals_in_set - inherited
    assert v.validation_info[16].thresholds.warn_at == 1
    assert v.validation_info[16].thresholds.stop_at == 2
    assert v.validation_info[16].thresholds.notify_at == 3

    # col_vals_in_set - overridden
    assert v.validation_info[17].thresholds.warn_at == 0.5
    assert v.validation_info[17].thresholds.stop_at is None
    assert v.validation_info[17].thresholds.notify_at is None

    # col_vals_not_in_set - inherited
    assert v.validation_info[18].thresholds.warn_at == 1
    assert v.validation_info[18].thresholds.stop_at == 2
    assert v.validation_info[18].thresholds.notify_at == 3

    # col_vals_not_in_set - overridden
    assert v.validation_info[19].thresholds.warn_at == 0.5
    assert v.validation_info[19].thresholds.stop_at is None
    assert v.validation_info[19].thresholds.notify_at is None

    # col_vals_null - inherited
    assert v.validation_info[20].thresholds.warn_at == 1
    assert v.validation_info[20].thresholds.stop_at == 2
    assert v.validation_info[20].thresholds.notify_at == 3

    # col_vals_null - overridden
    assert v.validation_info[21].thresholds.warn_at == 0.5
    assert v.validation_info[21].thresholds.stop_at is None
    assert v.validation_info[21].thresholds.notify_at is None

    # col_vals_not_null - inherited
    assert v.validation_info[22].thresholds.warn_at == 1
    assert v.validation_info[22].thresholds.stop_at == 2
    assert v.validation_info[22].thresholds.notify_at == 3

    # col_vals_not_null - overridden
    assert v.validation_info[23].thresholds.warn_at == 0.5
    assert v.validation_info[23].thresholds.stop_at is None
    assert v.validation_info[23].thresholds.notify_at is None

    # col_exists - inherited
    assert v.validation_info[24].thresholds.warn_at == 1
    assert v.validation_info[24].thresholds.stop_at == 2
    assert v.validation_info[24].thresholds.notify_at == 3

    # col_exists - overridden
    assert v.validation_info[25].thresholds.warn_at == 0.5
    assert v.validation_info[25].thresholds.stop_at is None
    assert v.validation_info[25].thresholds.notify_at is None


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_gt(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_gt(columns="x", value=0).interrogate().n_passed(i=1)[1] == 3
    assert (
        Validate(tbl).col_vals_gt(columns="x", value=0, na_pass=True).interrogate().n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_lt(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_lt(columns="x", value=10).interrogate().n_passed(i=1)[1] == 3
    assert (
        Validate(tbl)
        .col_vals_lt(columns="x", value=10, na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_eq(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_eq(columns="z", value=8).interrogate().n_passed(i=1)[1] == 3
    assert (
        Validate(tbl).col_vals_eq(columns="z", value=8, na_pass=True).interrogate().n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_ne(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_ne(columns="z", value=7).interrogate().n_passed(i=1)[1] == 3
    assert (
        Validate(tbl).col_vals_ne(columns="z", value=7, na_pass=True).interrogate().n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_ge(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_ge(columns="x", value=1).interrogate().n_passed(i=1)[1] == 3
    assert (
        Validate(tbl).col_vals_ge(columns="x", value=1, na_pass=True).interrogate().n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_le(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_le(columns="x", value=4).interrogate().n_passed(i=1)[1] == 3
    assert (
        Validate(tbl).col_vals_le(columns="x", value=4, na_pass=True).interrogate().n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_between(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl).col_vals_between(columns="x", left=1, right=4).interrogate().n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="x", left=11, right=14, na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="x", left=11, right=14, na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, inclusive=(False, True), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, inclusive=(True, False), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, inclusive=(False, False), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, inclusive=(False, False), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_outside(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl).col_vals_outside(columns="x", left=5, right=8).interrogate().n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="x", left=5, right=8, na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl).col_vals_outside(columns="x", left=4, right=8).interrogate().n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl).col_vals_outside(columns="x", left=-4, right=1).interrogate().n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="x", left=1, right=4, inclusive=(True, True))
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="x", left=1, right=4, inclusive=(True, True), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="x", left=4, right=8, inclusive=(False, True))
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="x", left=-4, right=1, inclusive=(True, False))
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="x", left=1, right=4, inclusive=(False, False), na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="x", left=1, right=4, inclusive=(False, False), na_pass=False)
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_col_vals_in_set(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl).col_vals_in_set(columns="x", set=[1, 2, 3, 4]).interrogate().n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[0, 1, 2, 3, 4, 5, 6])
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[1.0, 2.0, 3.0, 4.0])
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[1.00001, 2.00001, 3.00001, 4.00001])
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[-1, -2, -3, -4])
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_col_vals_not_in_set(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl).col_vals_not_in_set(columns="x", set=[5, 6, 7]).interrogate().n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[0, 1, 2, 3, 4, 5, 6])
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[1.0, 2.0, 3.0, 4.0])
        .interrogate()
        .n_passed(i=1)[1]
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[1.00001, 2.00001, 3.00001, 4.00001])
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[-1, -2, -3, -4])
        .interrogate()
        .n_passed(i=1)[1]
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_col_vals_regex(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_regex(columns="text", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
        .interrogate()
        .n_passed(i=1)[1]
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_regex(columns="text", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}", na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_regex(columns="text", pattern=r"^[0-9]-[a-z]{3}-[0-9]{3}$", na_pass=True)
        .interrogate()
        .n_passed(i=1)[1]
        == 3
    )


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_col_vals_null(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_null(columns="text").interrogate().n_passed(i=1)[1] == 1


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_col_vals_not_null(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_not_null(columns="text").interrogate().n_passed(i=1)[1] == 2


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_col_exists(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_exists(columns="text").interrogate().n_passed(i=1)[1] == 1
    assert Validate(tbl).col_exists(columns="invalid").interrogate().n_passed(i=1)[1] == 0


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_types(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    # Check that the `validation` object is a Validate object
    assert isinstance(validation, Validate)

    # Check that using the `get_tabular_report()` returns a GT object
    assert isinstance(validation.get_tabular_report(), GT.GT)


def test_load_dataset():

    # Load the default dataset (`small_table`) and verify it's a Polars DataFrame
    tbl = load_dataset()

    assert isinstance(tbl, pl.DataFrame)

    # Load the default dataset (`small_table`) and verify it's a Pandas DataFrame
    tbl = load_dataset(tbl_type="pandas")

    assert isinstance(tbl, pd.DataFrame)

    # Load the `game_revenue` dataset and verify it's a Polars DataFrame
    tbl = load_dataset(dataset="game_revenue")

    assert isinstance(tbl, pl.DataFrame)

    # Load the `game_revenue` dataset and verify it's a Pandas DataFrame

    tbl = load_dataset(dataset="game_revenue", tbl_type="pandas")

    assert isinstance(tbl, pd.DataFrame)


def test_load_dataset_invalid():

    # A ValueError is raised when an invalid dataset name is provided
    with pytest.raises(ValueError):
        load_dataset(dataset="invalid_dataset")

    # A ValueError is raised when an invalid table type is provided
    with pytest.raises(ValueError):
        load_dataset(tbl_type="invalid_tbl_type")


def test_load_dataset_no_pandas():

    # Mock the absence of the pandas library
    with patch.dict(sys.modules, {"pandas": None}):
        # A ValueError is raised when `tbl_type="pandas"` and the `pandas` package is not installed
        with pytest.raises(ImportError):
            load_dataset(tbl_type="pandas")


def test_load_dataset_no_polars():

    # Mock the absence of the polars library
    with patch.dict(sys.modules, {"polars": None}):
        # A ValueError is raised when `tbl_type="pandas"` and the `pandas` package is not installed
        with pytest.raises(ImportError):
            load_dataset(tbl_type="polars")
