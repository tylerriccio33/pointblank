from __future__ import annotations

import pytest

from pointblank._comparison import Comparator


@pytest.mark.parametrize(
    "x,compare,low,high,x_out,compare_out,low_out,high_out",
    [
        # `x`, and `compare` as lists of same length; `low` and `high` as `None`
        (
            # in
            [2, 3, 4],
            [3, 2, 7.4],
            None,
            None,
            # out
            [2, 3, 4],
            [3, 2, 7.4],
            None,
            None,
        ),
        # `x`, `compare`, `low` and `high` as scalars promoted to lists
        (
            # in
            3.2,
            4.2,
            1.2,
            3.2,
            # out
            [3.2],
            [4.2],
            [1.2],
            [3.2],
        ),
        # `x` as list, scalar `compare`, `low` and `high` values expressed as list matching the
        # length of `x`
        (
            # in
            [6.3, 2.1, 3.2, 4.2],
            8.3,
            1.6,
            10.5,
            # out
            [6.3, 2.1, 3.2, 4.2],
            [8.3, 8.3, 8.3, 8.3],
            [1.6, 1.6, 1.6, 1.6],
            [10.5, 10.5, 10.5, 10.5],
        ),
    ],
)
def test_comparison_class(
    x: float | int | list[float | int],
    compare: float | int | list[float | int] | None,
    low: float | int | list[float | int] | None,
    high: float | int | list[float | int] | None,
    x_out: list[float | int],
    compare_out: list[float | int],
    low_out: list[float | int] | None,
    high_out: list[float | int] | None,
):

    comp = Comparator(x=x, compare=compare, low=low, high=high)

    assert comp.x == x_out
    assert comp.compare == compare_out
    assert comp.low == low_out
    assert comp.high == high_out
