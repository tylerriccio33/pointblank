from __future__ import annotations

import pytest

from pointblank.thresholds import (
    Thresholds,
    _convert_abs_count_to_fraction,
    _normalize_thresholds_creation,
    _threshold_check,
)


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


@pytest.mark.parametrize(
    "param, value",
    [
        ("warn_at", -1),
        ("warn_at", -0.1),
        ("stop_at", -1),
        ("stop_at", -0.1),
        ("notify_at", -1),
        ("notify_at", -0.1),
    ],
)
def test_thresholds_raises_on_negative(param, value):
    with pytest.raises(ValueError):
        Thresholds(**{param: value})


def test_thresolds_repr():
    t = Thresholds(warn_at=1, stop_at=2, notify_at=3)
    assert repr(t) == "Thresholds(warn_at=1, stop_at=2, notify_at=3)"
    assert str(t) == "Thresholds(warn_at=1, stop_at=2, notify_at=3)"


@pytest.mark.parametrize("level", ["warn", "stop", "notify"])
def test_threshold_get_default(level):
    t = Thresholds()
    assert t._get_threshold_value(level=level) is None


@pytest.mark.parametrize("level", ["warn", "stop", "notify"])
def test_threshold_get_zero(level):
    t = Thresholds(warn_at=0, stop_at=0, notify_at=0)
    assert t._get_threshold_value(level=level) == 0


@pytest.mark.parametrize(
    "level, expected_value",
    [
        ("warn", 1),
        ("stop", 2),
        ("notify", 3),
    ],
)
def test_threshold_get_absolute(level, expected_value):
    t = Thresholds(warn_at=1, stop_at=2, notify_at=3)
    assert t._get_threshold_value(level=level) == expected_value


@pytest.mark.parametrize(
    "level, value",
    [
        ("warn", 0.1),
        ("stop", 0.2),
        ("notify", 0.3),
    ],
)
def test_threshold_get_fractional_0_1(level, value):
    t = Thresholds(warn_at=0.1, stop_at=0.2, notify_at=0.3)
    assert t._get_threshold_value(level=level) == value


@pytest.mark.parametrize(
    "fraction_failing, test_units, level, expected",
    [
        (0.1, 100, "warn", False),
        (0.25, 100, "warn", True),
        (0.3, 100, "warn", True),
        (0.4, 100, "stop", False),
        (0.5, 100, "stop", True),
        (0.6, 100, "stop", True),
        (0.74, 100, "notify", False),
        (0.75, 100, "notify", True),
        (0.76, 100, "notify", True),
    ],
)
def test_threshold_result_fractional(fraction_failing, test_units, level, expected):
    t = Thresholds(warn_at=0.25, stop_at=0.5, notify_at=0.75)
    assert (
        t._threshold_result(fraction_failing=fraction_failing, test_units=test_units, level=level)
        == expected
    )


@pytest.mark.parametrize(
    "fraction_failing, test_units, level, expected",
    [
        (0.1, 100, "warn", False),
        (0.25, 100, "warn", True),
        (0.3, 100, "warn", True),
        (0.4, 100, "stop", False),
        (0.5, 100, "stop", True),
        (0.6, 100, "stop", True),
        (0.74, 100, "notify", False),
        (0.75, 100, "notify", True),
        (0.76, 100, "notify", True),
    ],
)
def test_threshold_result_absolute(fraction_failing, test_units, level, expected):
    t = Thresholds(warn_at=25, stop_at=50, notify_at=75)
    assert (
        t._threshold_result(fraction_failing=fraction_failing, test_units=test_units, level=level)
        == expected
    )


@pytest.mark.parametrize(
    "level",
    ["warn", "stop", "notify"],
)
def test_threshold_result_zero(level):
    t = Thresholds(warn_at=0, stop_at=0, notify_at=0)

    assert t._threshold_result(fraction_failing=0, test_units=100, level=level) is True
    assert t._threshold_result(fraction_failing=0.1, test_units=100, level=level) is True
    assert t._threshold_result(fraction_failing=1.5, test_units=100, level=level) is True


@pytest.mark.parametrize(
    "level",
    ["warn", "stop", "notify"],
)
def test_threshold_result_none(level):
    t = Thresholds()

    assert t._threshold_result(fraction_failing=0, test_units=100, level=level) is None
    assert t._threshold_result(fraction_failing=0.1, test_units=100, level=level) is None
    assert t._threshold_result(fraction_failing=1.5, test_units=100, level=level) is None


@pytest.mark.parametrize(
    "value, test_units, expected",
    [
        (None, 100, None),
        (1, 100, 0.01),
        (1, 1e7, 1e-07),
        (1, 1e10, 1e-10),
        (0, 100, 0.0),
        (1.6, 10, 0.2),
        (1.4, 10, 0.1),
        (100, 100, 1.0),
        (150, 100, 1.5),
    ],
)
def test_convert_abs_count_to_fraction(value, test_units, expected):
    assert _convert_abs_count_to_fraction(value=value, test_units=test_units) == expected


@pytest.mark.parametrize(
    "value, test_units",
    [
        (0, 100),
    ],
)
def test_convert_abs_count_to_fraction_type(value, test_units):
    assert isinstance(_convert_abs_count_to_fraction(value=value, test_units=test_units), float)


def test_convert_abs_count_to_fraction_raises():

    with pytest.raises(ValueError):
        _convert_abs_count_to_fraction(value=-1, test_units=100)
    with pytest.raises(ValueError):
        _convert_abs_count_to_fraction(value=-0.1, test_units=100)
    with pytest.raises(ValueError):
        _convert_abs_count_to_fraction(value=3, test_units=-100)
    with pytest.raises(ValueError):
        _convert_abs_count_to_fraction(value=3, test_units=0)


def test_normalize_thresholds_creation():

    # None should be equivalent to the default Thresholds object
    assert _normalize_thresholds_creation(thresholds=None) == Thresholds()

    # Use of integers or floats should be equivalent, applying the value to the `warn_at` attribute
    assert _normalize_thresholds_creation(thresholds=1) == Thresholds(
        warn_at=1, stop_at=None, notify_at=None
    )
    assert _normalize_thresholds_creation(thresholds=0.5) == Thresholds(
        warn_at=0.5, stop_at=None, notify_at=None
    )

    # Use of a tuple will vary depending on the length of the tuple
    assert _normalize_thresholds_creation(thresholds=(1,)) == Thresholds(
        warn_at=1, stop_at=None, notify_at=None
    )
    assert _normalize_thresholds_creation(thresholds=(0.2, 20)) == Thresholds(
        warn_at=0.2, stop_at=20, notify_at=None
    )
    assert _normalize_thresholds_creation(thresholds=(0.2, 20, 0.5)) == Thresholds(
        warn_at=0.2, stop_at=20, notify_at=0.5
    )

    # ...but the tuple should have 1-3 elements, otherwise it should raise an error
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds=())

    # Use of a dictionary should convert the dictionary to a Thresholds object
    assert _normalize_thresholds_creation(thresholds={"warn_at": 1}) == Thresholds(
        warn_at=1, stop_at=None, notify_at=None
    )
    assert _normalize_thresholds_creation(thresholds={"warn_at": 0.2, "stop_at": 20}) == Thresholds(
        warn_at=0.2, stop_at=20, notify_at=None
    )
    assert _normalize_thresholds_creation(
        thresholds={"warn_at": 0.2, "stop_at": 20, "notify_at": 0.5}
    ) == Thresholds(warn_at=0.2, stop_at=20, notify_at=0.5)
    assert _normalize_thresholds_creation(thresholds={}) == Thresholds()

    # ...but the dictionary keys need to be valid Thresholds attributes
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(
            thresholds={"warn_at": 1, "stop_at": 2, "notify_at": 3, "extra": 4}
        )
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds={"warn_at": 1, "stop_at": 2, "invalid": 3})
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds={"warn": 1})

    # Use of a Thresholds object should return the object as is
    assert _normalize_thresholds_creation(thresholds=Thresholds()) == Thresholds()
    assert _normalize_thresholds_creation(
        thresholds=Thresholds(warn_at=1, notify_at=0.5)
    ) == Thresholds(warn_at=1, notify_at=0.5)

    # Anything else should raise an error
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds="not a valid value")
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds=1.0 + 1j)
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds=[])
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds=[0.2])
    with pytest.raises(ValueError):
        _normalize_thresholds_creation(thresholds=object())


def test_threshold_check():

    assert _threshold_check(failing_test_units=6, threshold=5) is False
    assert _threshold_check(failing_test_units=5, threshold=5) is False
    assert _threshold_check(failing_test_units=4, threshold=5) is True
    assert _threshold_check(failing_test_units=0, threshold=0) is False
    assert _threshold_check(failing_test_units=80, threshold=None) is False
