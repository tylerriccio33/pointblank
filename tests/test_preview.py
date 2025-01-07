import pytest
import sys

from unittest.mock import patch

from pointblank.column import (
    col,
    starts_with,
    ends_with,
    contains,
    matches,
    everything,
    first_n,
    last_n,
)
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


def test_preview_no_polars_duckdb_table():

    small_table = load_dataset(dataset="small_table", tbl_type="duckdb")

    # Mock the absence of the Polars library, which is the default library for making
    # a table for the preview; this should not raise an error since Pandas is the
    # fallback library and is available
    with patch.dict(sys.modules, {"polars": None}):
        preview(small_table)

    # Mock the absence of the Pandas library, which is a secondary library for making
    # a table for the preview; this should not raise an error since Polars is the default
    # library and is available
    with patch.dict(sys.modules, {"pandas": None}):
        preview(small_table)

    # Mock the absence of both the Polars and Pandas libraries, which are the libraries
    # for making a table for the preview; this should raise an error since there are no
    # libraries available to make a table for the preview
    with patch.dict(sys.modules, {"polars": None, "pandas": None}):
        with pytest.raises(ImportError):
            preview(small_table)


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_preview_with_columns_subset_no_fail(tbl_type):

    tbl = load_dataset(dataset="game_revenue", tbl_type=tbl_type)

    preview(tbl, columns_subset="player_id")
    preview(tbl, columns_subset=["player_id"])
    preview(tbl, columns_subset=["player_id", "item_name", "item_revenue"])
    preview(tbl, columns_subset=col("player_id"))
    preview(tbl, columns_subset=col(matches("player_id")))
    preview(tbl, columns_subset=col(matches("_id")))
    preview(tbl, columns_subset=starts_with("item"))
    preview(tbl, columns_subset=ends_with("revenue"))
    preview(tbl, columns_subset=matches("_id"))
    preview(tbl, columns_subset=contains("_"))
    preview(tbl, columns_subset=everything())
    preview(tbl, columns_subset=col(starts_with("item") | matches("player")))
    preview(tbl, columns_subset=col(first_n(2) | last_n(2)))
    preview(tbl, columns_subset=col(everything() - last_n(2)))
    preview(tbl, columns_subset=col(~first_n(2)))


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_preview_with_columns_subset_failing(tbl_type):

    tbl = load_dataset(dataset="game_revenue", tbl_type=tbl_type)

    with pytest.raises(ValueError):
        preview(tbl, columns_subset="player_id", n_head=100, n_tail=100)
    with pytest.raises(ValueError):
        preview(tbl, columns_subset="fake_id")
    with pytest.raises(ValueError):
        preview(tbl, columns_subset=["fake_id", "item_name", "item_revenue"])
    with pytest.raises(ValueError):
        preview(tbl, columns_subset=col(matches("fake_id")))
