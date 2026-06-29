import numpy as np
import pytest

from pyresiflex.generator.generator_complex_impedance import (
    GaussianGenerator,
    TrapezoidalGenerator,
)
from tests.helpers import relative_close


def test_trapezoidal_generator_voltage_off_before_start() -> None:
    """Check the trapezoidal voltage is at ``U_off`` before the pulse."""
    gen = TrapezoidalGenerator(
        U_off=1.0, U_on=5.0, t_rise=2.0, t_on=4.0, t_fall=2.0
    )
    # Before pulse starts
    assert np.isclose(gen.generator_voltage(-1.0), 1.0)


def test_trapezoidal_generator_voltage_rising() -> None:
    """Verify the linear rising edge of the trapezoidal voltage.

    Check the voltage interpolates linearly from ``U_off`` to ``U_on``
    during ``t_rise``.
    """
    gen = TrapezoidalGenerator(
        U_off=0.0, U_on=10.0, t_rise=2.0, t_on=4.0, t_fall=2.0
    )
    # During rising edge
    t = 1.0
    expected = 0.0 + (10.0 - 0.0) * t / 2.0
    assert np.isclose(gen.generator_voltage(t), expected)


def test_trapezoidal_generator_voltage_plateau() -> None:
    """Check the trapezoidal voltage holds at ``U_on`` on the plateau."""
    gen = TrapezoidalGenerator(
        U_off=0.0, U_on=10.0, t_rise=2.0, t_on=4.0, t_fall=2.0
    )
    # During plateau
    t = 3.0
    assert np.isclose(gen.generator_voltage(t), 10.0)
    t = 5.0
    assert np.isclose(gen.generator_voltage(t), 10.0)


def test_trapezoidal_generator_voltage_falling() -> None:
    """Verify the linear falling edge of the trapezoidal voltage.

    Check the voltage interpolates linearly from ``U_on`` back to
    ``U_off`` during ``t_fall``.
    """
    gen = TrapezoidalGenerator(
        U_off=0.0, U_on=10.0, t_rise=2.0, t_on=4.0, t_fall=2.0
    )
    # During falling edge
    t = 6.0  # t_rise + t_on = 6.0
    expected = 10.0 - (10.0 - 0.0) * (t - 2.0 - 4.0) / 2.0
    assert np.isclose(gen.generator_voltage(t), expected)
    t = 7.0
    expected = 10.0 - (10.0 - 0.0) * (t - 2.0 - 4.0) / 2.0
    assert np.isclose(gen.generator_voltage(t), expected)


def test_trapezoidal_generator_voltage_after_pulse() -> None:
    """Check the trapezoidal voltage returns to ``U_off`` after the pulse."""
    gen = TrapezoidalGenerator(
        U_off=2.0, U_on=8.0, t_rise=1.0, t_on=2.0, t_fall=1.0
    )
    # After pulse ends
    t = 5.0  # t_rise + t_on + t_fall = 4.0
    assert np.isclose(gen.generator_voltage(t), 2.0)


def test_trapezoidal_generator_impedance_constant() -> None:
    """Check the trapezoidal impedance equals ``R_g`` at every frequency."""
    gen = TrapezoidalGenerator(R_g=15.0)
    freq = np.array([0.0, 1.0, 10.0, 1e6])
    imp = gen.generator_impedance(freq)
    assert np.allclose(imp, 15.0)


def test_gaussian_generator_voltage_peak() -> None:
    """Check the gaussian voltage equals ``height`` at the mean (peak)."""
    height = 5.0
    mean = 1e-6
    FWHM = 2e-6
    gen = GaussianGenerator(height=height, mean=mean, FWHM=FWHM)
    # At mean, voltage should be at peak (height)
    assert np.isclose(gen.generator_voltage(mean), height)


def test_gaussian_generator_voltage_far_from_mean() -> None:
    """Check the gaussian voltage decays to near zero far from the mean."""
    gen = GaussianGenerator(height=5.0, mean=0.0, FWHM=1.0)
    # Far from mean, voltage should be close to zero
    assert gen.generator_voltage(100.0) < 1e-6


def test_gaussian_generator_impedance_ideal() -> None:
    """Check an ideal source (no ``R_g``/``C_g``) has zero impedance."""
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0)
    freq = np.array([0.0, 1.0, 10.0])
    imp = gen.generator_impedance(freq)
    assert np.allclose(imp, 0.0)


def test_gaussian_generator_impedance_resistive() -> None:
    """Check a resistive gaussian source has impedance ``R_g`` at all f."""
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=10.0)
    freq = np.array([0.0, 1.0, 10.0])
    imp = gen.generator_impedance(freq)
    assert np.allclose(imp, 10.0)


def test_gaussian_generator_impedance_resistive_capacitive() -> None:
    """Verify the RC source impedance is ``R_g`` at DC and rolls off.

    Check the impedance equals ``R_g`` at zero frequency and that its
    magnitude decreases at high frequency.
    """
    R_g = 10.0
    C_g = 1e-9
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=R_g, C_g=C_g)
    freq = np.array([0.0, 1e6])
    imp = gen.generator_impedance(freq)
    # At zero frequency, impedance is R_g
    assert np.isclose(imp[0], R_g)
    # At high frequency, impedance should decrease
    assert np.abs(imp[1]) < R_g


def test_gaussian_generator_pure_capacitance() -> None:
    """Verify the pure-capacitance branch is infinite at DC.

    A purely capacitive source (``R_g == 0``, ``C_g != 0``) follows
    ``Z_g(f) = 1 / (j 2 pi f C_g)``: infinite at DC and matching the
    ideal-capacitor impedance at non-zero frequencies.
    """
    C_g = 1e-9
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=0.0, C_g=C_g)
    freq = np.array([0.0, 1e3, 1e6])
    imp = gen.generator_impedance(freq)
    # Infinite impedance at zero frequency (the capacitor blocks DC).
    assert np.isinf(np.abs(imp[0]))
    # Matches the ideal-capacitor impedance at non-zero frequency.
    expected = 1 / (1j * 2 * np.pi * freq[1:] * C_g)
    assert relative_close(imp[1:], expected, 1e-12)


@pytest.mark.parametrize("R_g, C_g", [(-1.0, 0.0), (0.0, -1e-9)])
def test_gaussian_generator_rejects_negative_components(
    R_g: float, C_g: float
) -> None:
    """Check negative generator resistance or capacitance is rejected."""
    with pytest.raises(ValueError, match="must be >= 0"):
        GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=R_g, C_g=C_g)


def test_gaussian_function_shape() -> None:
    """Verify the gaussian static method peaks and halves correctly.

    Check the value equals ``height`` at the mean and is exactly half
    the height at ``mean + FWHM/2`` (the full-width-at-half-maximum
    definition).
    """
    # Test the gaussian static method directly
    t = 0.0
    height = 2.0
    mean = 0.0
    FWHM = 1.0
    val = GaussianGenerator.gaussian(t, height, mean, FWHM)
    assert relative_close(val, height, 1e-12)
    # At t = mean + FWHM/2 the value is exactly half the height (by the
    # definition of the full width at half maximum).
    t_half = mean + FWHM / 2
    val_half = GaussianGenerator.gaussian(t_half, height, mean, FWHM)
    assert relative_close(val_half, height / 2, 1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
