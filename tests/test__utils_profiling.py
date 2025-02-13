import pytest

from pointblank.validate import load_dataset
from pointblank._utils_profiling import DataProfiler


def test_data_profiler_class():

    dataset = load_dataset(dataset="small_table")
    profiler = DataProfiler(data=dataset)

    assert profiler.tbl_name is None
    assert profiler.tbl_category == "dataframe"
    assert profiler.tbl_type == "polars"
    assert profiler.data_native.equals(dataset)
    assert profiler.profile is not None


def test_data_profiler_class_raises():
    with pytest.raises(TypeError):
        DataProfiler(data="not a DataFrame or Ibis Table")

    with pytest.raises(TypeError):
        DataProfiler(data=123)

    with pytest.raises(TypeError):
        DataProfiler(data=[1, 2, 3])
