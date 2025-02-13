import pytest

from pointblank.validate import load_dataset
from pointblank._utils_profiling import DataProfiler


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_data_profiler_class(tbl_type):

    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    profiler = DataProfiler(data=dataset)

    assert profiler.data.equals(dataset)
    assert profiler.tbl_name is None
    assert profiler.profile is not None
    assert isinstance(profiler.profile, dict)

    if tbl_type == "duckdb":
        assert profiler.tbl_type == "duckdb"
        assert profiler.tbl_category == "ibis"
        assert profiler.data_alt is None

    if tbl_type == "polars":
        assert profiler.tbl_type == "polars"
        assert profiler.tbl_category == "dataframe"
        assert profiler.data_alt is not None

    if tbl_type == "pandas":
        assert profiler.tbl_type == "pandas"
        assert profiler.tbl_category == "dataframe"
        assert profiler.data_alt is not None


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_data_profiler_dict_output(tbl_type):

    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    profiler = DataProfiler(data=dataset)

    assert isinstance(profiler.profile, dict)

    profile_dict = profiler.get_profile()

    assert isinstance(profile_dict, dict)

    assert profiler.profile == profile_dict


def test_data_profiler_json_output():

    dataset = load_dataset(dataset="small_table")
    profiler = DataProfiler(data=dataset)

    profile_json = profiler.to_json()

    assert isinstance(profile_json, str)


def test_data_profiler_json_file_output(tmp_path):

    dataset = load_dataset(dataset="small_table")
    profiler = DataProfiler(data=dataset)

    profile_json = profiler.to_json()

    file_path = tmp_path / "profile.json"
    profiler.save_to_json(output_file=file_path)

    assert file_path.exists()
    assert file_path.is_file()

    with open(file_path, "r") as f:
        file_content = f.read()

    assert profile_json == file_content


def test_data_profiler_class_raises():
    with pytest.raises(TypeError):
        DataProfiler(data="not a DataFrame or Ibis Table")

    with pytest.raises(TypeError):
        DataProfiler(data=123)

    with pytest.raises(TypeError):
        DataProfiler(data=[1, 2, 3])
