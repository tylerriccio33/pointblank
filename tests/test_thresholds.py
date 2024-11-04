from __future__ import annotations

import pytest

from pointblank.thresholds import Thresholds


def test_thresholds():
    t = Thresholds(warn_at=1, stop_at=2, notify_at=3)

    assert t.warn_fraction is None
    assert t.warn_count == 1

    assert t.stop_fraction is None
    assert t.stop_count == 2

    assert t.notify_fraction is None
    assert t.notify_count == 3

    with pytest.raises(ValueError):
        Thresholds(warn_at=-1)

    with pytest.raises(ValueError):
        Thresholds(stop_at=-0.1)

    with pytest.raises(ValueError):
        Thresholds(notify_at=-20.5)
