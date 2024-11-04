from __future__ import annotations

import pytest

from pointblank.thresholds import Thresholds, _convert_abs_count_to_fraction


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


def test_thresolds_repr():
    t = Thresholds(warn_at=1, stop_at=2, notify_at=3)
    assert repr(t) == "Thresholds(warn_at=1, stop_at=2, notify_at=3)"
    assert str(t) == "Thresholds(warn_at=1, stop_at=2, notify_at=3)"


def test_threshold_get_default():
    t = Thresholds()

    assert t._get_threshold_value("warn") is None
    assert t._get_threshold_value("stop") is None
    assert t._get_threshold_value("notify") is None


def test_threshold_get_zero():
    t = Thresholds(warn_at=0, stop_at=0, notify_at=0)

    assert t._get_threshold_value("warn") == 0
    assert t._get_threshold_value("stop") == 0
    assert t._get_threshold_value("notify") == 0


def test_threshold_get_absolute():
    t = Thresholds(warn_at=1, stop_at=2, notify_at=3)

    assert t._get_threshold_value("warn") == 1
    assert t._get_threshold_value("stop") == 2
    assert t._get_threshold_value("notify") == 3


def test_threshold_get_fractional_0_1():
    t = Thresholds(warn_at=0.1, stop_at=0.2, notify_at=0.3)

    assert t._get_threshold_value("warn") == 0.1
    assert t._get_threshold_value("stop") == 0.2
    assert t._get_threshold_value("notify") == 0.3


def test_convert_abs_count_to_fraction():

    assert _convert_abs_count_to_fraction(None, 100) == None
    assert _convert_abs_count_to_fraction(1, 100) == 0.01
    assert _convert_abs_count_to_fraction(1, 1e7) == 1e-07
    assert _convert_abs_count_to_fraction(1, 1e10) == 1e-10
    assert _convert_abs_count_to_fraction(0, 100) == 0.0
    assert _convert_abs_count_to_fraction(1.6, 10) == 0.2
    assert _convert_abs_count_to_fraction(1.4, 10) == 0.1
    assert isinstance(_convert_abs_count_to_fraction(0, 100), float)
    assert _convert_abs_count_to_fraction(100, 100) == 1.0
    assert _convert_abs_count_to_fraction(150, 100) == 1.5

    with pytest.raises(ValueError):
        _convert_abs_count_to_fraction(-1, 100)
    with pytest.raises(ValueError):
        _convert_abs_count_to_fraction(-0.1, 100)
    with pytest.raises(ValueError):
        _convert_abs_count_to_fraction(3, -100)
    with pytest.raises(ValueError):
        _convert_abs_count_to_fraction(3, 0)
