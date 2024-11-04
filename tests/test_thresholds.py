from __future__ import annotations

import pytest

from pointblank.thresholds import Thresholds


def test_thresholds_default():
    t = Thresholds()

    assert t.warn_fraction is None
    assert t.warn_count is None

    assert t.stop_fraction is None
    assert t.stop_count is None

    assert t.notify_fraction is None
    assert t.notify_count is None


def test_thresholds_absolute():
    t = Thresholds(warn_at=1, stop_at=2, notify_at=3)

    assert t.warn_fraction is None
    assert t.warn_count == 1

    assert t.stop_fraction is None
    assert t.stop_count == 2

    assert t.notify_fraction is None
    assert t.notify_count == 3


def test_thresholds_fractional_0_1():
    t = Thresholds(warn_at=0.01, stop_at=0.1464, notify_at=0.9999)

    assert t.warn_fraction == 0.01
    assert t.warn_count is None

    assert t.stop_fraction == 0.1464
    assert t.stop_count is None

    assert t.notify_fraction == 0.9999
    assert t.notify_count is None


def test_thresholds_zero():
    t = Thresholds(warn_at=0, stop_at=0, notify_at=0)

    assert t.warn_fraction == 0
    assert t.warn_count == 0

    assert t.stop_fraction == 0
    assert t.stop_count == 0

    assert t.notify_fraction == 0
    assert t.notify_count == 0


def test_thresholds_absolute_rounded():
    t = Thresholds(warn_at=1.4, stop_at=2.99, notify_at=4.5)

    assert t.warn_fraction is None
    assert t.warn_count == 1

    assert t.stop_fraction is None
    assert t.stop_count == 3

    assert t.notify_fraction is None
    assert t.notify_count == 4


def test_thresholds_raises_on_negative():
    t = Thresholds(warn_at=1, stop_at=2, notify_at=3)

    with pytest.raises(ValueError):
        Thresholds(warn_at=-1)
    with pytest.raises(ValueError):
        Thresholds(warn_at=-0.1)

    with pytest.raises(ValueError):
        Thresholds(stop_at=-1)
    with pytest.raises(ValueError):
        Thresholds(stop_at=-0.1)

    with pytest.raises(ValueError):
        Thresholds(notify_at=-1)
    with pytest.raises(ValueError):
        Thresholds(notify_at=-0.1)
