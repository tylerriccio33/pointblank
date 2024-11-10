import pytest
import pandas as pd
import polars as pl

from pointblank.validate import Validate
from pointblank.thresholds import Thresholds


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pd():
    return pd.DataFrame({"x": [1, 2, pd.NA, 4], "y": [4, pd.NA, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pl():
    return pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, 8, 8, 8]})


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_col_vals_all_passing(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(column="x", value=0).interrogate()

    assert v.data.shape == (4, 3)
    assert str(v.data["x"].dtype).lower() == "int64"
    assert str(v.data["y"].dtype).lower() == "int64"
    assert str(v.data["z"].dtype).lower() == "int64"

    # There is a single validation check entry in the `validation_info` attribute
    assert len(v.validation_info) == 1

    # The single step had no failing test units so the `all_passed` attribute is `True`
    assert v.all_passed()

    # Test other validation types for all passing behavior in single steps
    assert Validate(tbl).col_vals_lt(column="x", value=5).interrogate().all_passed()
    assert Validate(tbl).col_vals_eq(column="z", value=8).interrogate().all_passed()
    assert Validate(tbl).col_vals_ge(column="x", value=1).interrogate().all_passed()
    assert Validate(tbl).col_vals_le(column="x", value=4).interrogate().all_passed()
    assert Validate(tbl).col_vals_between(column="x", left=0, right=5).interrogate().all_passed()
    assert Validate(tbl).col_vals_outside(column="x", left=-5, right=0).interrogate().all_passed()
    assert Validate(tbl).col_vals_in_set(column="x", set=[1, 2, 3, 4, 5]).interrogate().all_passed()
    assert Validate(tbl).col_vals_not_in_set(column="x", set=[5, 6, 7]).interrogate().all_passed()


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_plan(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Create a validation plan
    v = Validate(tbl).col_vals_gt(column="x", value=0)

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
        "row_sample",
        "tbl_checked",
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
    assert val_info.row_sample is None
    assert val_info.tbl_checked is None
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
        "row_sample",
        "tbl_checked",
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
    assert val_info.row_sample is None
    assert val_info.tbl_checked is True
    assert isinstance(val_info.time_processed, str)
    assert val_info.proc_duration_s > 0.0


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_attr_getters(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(column="x", value=0).interrogate()

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


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_report(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(column="x", value=0).interrogate()

    assert v.report_as_json() != v.report_as_json(
        exclude_fields=["time_processed", "proc_duration_s"]
    )

    # A ValueError is raised when `use_fields=` includes invalid fields
    with pytest.raises(ValueError):
        v.report_as_json(use_fields=["invalid_field"])

    # A ValueError is raised when `exclude_fields=` includes invalid fields
    with pytest.raises(ValueError):
        v.report_as_json(exclude_fields=["invalid_field"])

    # A ValueError is raised `use_fields=` and `exclude_fields=` are both provided
    with pytest.raises(ValueError):
        v.report_as_json(use_fields=["i"], exclude_fields=["i_o"])


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_report_interrogate_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(column="x", value=0)
        .interrogate()
        .report_as_json(exclude_fields=["time_processed", "proc_duration_s"])
    )

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(report, "validation_report.json")


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_report_no_interrogate_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(column="x", value=0)
        .report_as_json(exclude_fields=["time_processed", "proc_duration_s"])
    )

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(report, "validation_report.json")


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_report_use_fields_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(column="x", value=0)
        .report_as_json(
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


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_check_column_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `column=` is not a string
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(column=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(column=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(column=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(column=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(column=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(column=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(column=9, left=0, right=5)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(column=9, left=-5, right=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_in_set(column=9, set=[1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_not_in_set(column=9, set=[5, 6, 7])


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_check_na_pass_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `na_pass=` is not a boolean
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(column="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(column="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(column="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(column="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(column="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(column="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(column="x", left=0, right=5, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(column="x", left=-5, right=0, na_pass=9)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_validation_check_active_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `active=` is not a boolean
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(column="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(column="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(column="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(column="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(column="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(column="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(column="x", left=0, right=5, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(column="x", left=-5, right=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_in_set(column="x", set=[1, 2, 3, 4, 5], active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_not_in_set(column="x", set=[5, 6, 7], active=9)
