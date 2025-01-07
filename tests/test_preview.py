import pytest

from pointblank.preview import preview
from pointblank.validate import load_dataset


def test_preview_no_fail_pd_table():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")

    preview(small_table)
    preview(small_table, n_head=2)
    preview(small_table, n_tail=2)
    preview(small_table, n_head=2, n_tail=2)


def test_preview_no_fail_pl_table():

    small_table = load_dataset(dataset="small_table", tbl_type="polars")

    preview(small_table)
    preview(small_table, n_head=2)
    preview(small_table, n_tail=2)
    preview(small_table, n_head=2, n_tail=2)


def test_preview_no_fail_duckdb_table():

    small_table = load_dataset(dataset="small_table", tbl_type="duckdb")

    preview(small_table)
    preview(small_table, n_head=2)
    preview(small_table, n_tail=2)
    preview(small_table, n_head=2, n_tail=2)


def test_preview_large_head_tail_pd_table():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")
    preview(small_table, n_head=10, n_tail=10)


def test_preview_large_head_tail_pl_table():

    small_table = load_dataset(dataset="small_table", tbl_type="polars")
    preview(small_table, n_head=10, n_tail=10)


def test_preview_large_head_tail_duckdb_table():

    small_table = load_dataset(dataset="small_table", tbl_type="duckdb")
    preview(small_table, n_head=10, n_tail=10)


def test_preview_fails_head_tail_exceed_limit():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")

    with pytest.raises(ValueError):
        preview(small_table, n_head=100, n_tail=100)  # default limit is 50

    preview(small_table, n_head=100, n_tail=100, limit=300)
