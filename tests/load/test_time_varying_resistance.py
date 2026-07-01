import numpy as np
import pytest

from pyresiflex.load.time_varying_resistance import (
    ConstantResistance,
    PlasmaResistanceExponentialFall,
    PlasmaResistanceInterpolate,
    PlasmaResistanceLinearFall,
)


def test_constant_resistance() -> None:
    """Check a constant resistance returns R for any time value."""
    R = 42.0
    cr = ConstantResistance(R)
    for t in [-100, 0, 1, 100]:
        assert cr.load_impedance(t) == R


def test_plasma_resistance_linear_fall() -> None:
    """Verify a linear fall: held before/after, linear during ramp.

    The resistance stays at ``Z_start`` up to ``t_start``, falls
    linearly to ``Z_end`` over the ramp, then is held at ``Z_end``.
    """
    Z_start = 10.0
    Z_end = 5.0
    t_start = 0.0
    t_end = 1.0
    pr = PlasmaResistanceLinearFall(Z_start, Z_end, t_start, t_end)
    # Before start: held at the initial value.
    assert pr.load_impedance(-1.0) == Z_start
    assert pr.load_impedance(0.0) == Z_start
    # During ramp: midpoint is the average of start and end.
    assert pr.load_impedance(0.5) == 7.5
    # After end: held at the final value.
    assert pr.load_impedance(1.5) == Z_end
    assert pr.load_impedance(2.0) == Z_end


def test_plasma_resistance_exponential_fall() -> None:
    """Verify an exponential fall against the closed-form formula.

    The resistance is held before/after the ramp and follows the
    ``fall_exponent`` power law during it.
    """
    Z_start = 10.0
    Z_end = 5.0
    t_start = 0.0
    t_end = 1.0
    fall_exponent = 2.0
    pr = PlasmaResistanceExponentialFall(
        Z_start, Z_end, t_start, t_end, fall_exponent
    )
    # Before start: held at the initial value.
    assert pr.load_impedance(-1.0) == Z_start
    assert pr.load_impedance(0.0) == Z_start
    # During ramp: compare against the closed-form power-law value.
    t = 0.5
    expected = (
        Z_end
        + (Z_start - Z_end)
        * (1 - (t - t_start) / (t_end - t_start)) ** fall_exponent
    )
    assert pr.load_impedance(t) == expected
    # After end: held at the final value.
    assert pr.load_impedance(1.5) == Z_end
    assert pr.load_impedance(2.0) == Z_end


def test_plasma_resistance_interpolate() -> None:
    """Verify interpolated resistance: clamped ends, linear between.

    Outside the sample range the resistance is clamped to the first or
    last value; inside it is linearly interpolated.
    """
    t_array = np.array([0.0, 1.0, 2.0])
    R_array = np.array([10.0, 5.0, 7.0])
    pr = PlasmaResistanceInterpolate(t_array, R_array)
    # Before range: clamped to first value.
    assert pr.load_impedance(-1.0) == R_array[0]
    # At known points: exact sample values.
    assert pr.load_impedance(0.0) == 10.0
    assert pr.load_impedance(1.0) == 5.0
    assert pr.load_impedance(2.0) == 7.0
    # Between points: linear interpolation.
    assert pr.load_impedance(0.5) == 7.5
    assert pr.load_impedance(1.5) == 6.0
    # After range: clamped to last value.
    assert pr.load_impedance(3.0) == R_array[-1]


@pytest.mark.parametrize(
    "t_array, R_array",
    [
        (np.array([0.0, 1.0]), np.array([10.0, 5.0])),  # valid
        (np.array([0, 2]), np.array([1, 2])),  # valid
    ],
)
def test_check_arrays_valid(t_array: np.ndarray, R_array: np.ndarray) -> None:
    """Ensure ``check_arrays`` accepts valid time/resistance arrays."""
    PlasmaResistanceInterpolate(t_array, R_array)
    # Should not raise.


@pytest.mark.parametrize(
    "t_array, R_array, exc_type",
    [
        (
            [0.0, 1.0],
            np.array([10.0, 5.0]),
            TypeError,
        ),  # t_array not np.ndarray
        (
            np.array([0.0, 1.0]),
            [10.0, 5.0],
            TypeError,
        ),  # R_array not np.ndarray
        (
            np.array([[0.0, 1.0]]),
            np.array([10.0, 5.0]),
            ValueError,
        ),  # t_array not 1D
        (
            np.array([0.0, 1.0]),
            np.array([[10.0, 5.0]]),
            ValueError,
        ),  # R_array not 1D
        (
            np.array([0.0, 1.0]),
            np.array([10.0]),
            ValueError,
        ),  # length mismatch
        (np.array([0.0]), np.array([10.0]), ValueError),  # less than 2 points
        (
            np.array([1.0, 0.0]),
            np.array([10.0, 5.0]),
            ValueError,
        ),  # not sorted
        (
            np.array([0.0, 1.0]),
            np.array([10.0, np.nan]),
            ValueError,
        ),  # R_array nan
        (
            np.array([0.0, np.nan]),
            np.array([10.0, 5.0]),
            ValueError,
        ),  # t_array nan
        (
            np.array([0.0, np.inf]),
            np.array([10.0, 5.0]),
            ValueError,
        ),  # t_array inf
        (
            np.array([0.0, 1.0]),
            np.array([10.0, np.inf]),
            ValueError,
        ),  # R_array inf
        (
            np.array([0.0, 1.0]),
            np.array([10.0, 5.0j]),
            ValueError,
        ),  # R_array not real
        (
            np.array([0.0, 1.0j]),
            np.array([10.0, 5.0]),
            ValueError,
        ),  # t_array not real
    ],
)
def test_check_arrays_invalid_only(
    t_array: np.ndarray, R_array: np.ndarray, exc_type: type[Exception]
) -> None:
    """Check ``check_arrays`` rejects each invalid array case.

    Covers wrong type, wrong dimensions, length mismatch, too few
    points, unsorted times, and non-finite or complex values.
    """
    with pytest.raises(exc_type):
        PlasmaResistanceInterpolate(t_array, R_array)

    with pytest.raises(exc_type):
        PlasmaResistanceInterpolate.check_arrays(
            PlasmaResistanceInterpolate,  # type: ignore
            t_array,
            R_array,
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
