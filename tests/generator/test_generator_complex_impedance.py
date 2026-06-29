import numpy as np
import pytest

from pyresiflex.generator.generator_complex_impedance import (
    GaussianGenerator,
    TrapezoidalGenerator,
)


def test_trapezoidal_generator_voltage_off_before_start():
    gen = TrapezoidalGenerator(
        U_off=1.0, U_on=5.0, t_rise=2.0, t_on=4.0, t_fall=2.0
    )
    # Before pulse starts
    assert np.isclose(gen.generator_voltage(-1.0), 1.0)


def test_trapezoidal_generator_voltage_rising():
    gen = TrapezoidalGenerator(
        U_off=0.0, U_on=10.0, t_rise=2.0, t_on=4.0, t_fall=2.0
    )
    # During rising edge
    t = 1.0
    expected = 0.0 + (10.0 - 0.0) * t / 2.0
    assert np.isclose(gen.generator_voltage(t), expected)


def test_trapezoidal_generator_voltage_plateau():
    gen = TrapezoidalGenerator(
        U_off=0.0, U_on=10.0, t_rise=2.0, t_on=4.0, t_fall=2.0
    )
    # During plateau
    t = 3.0
    assert np.isclose(gen.generator_voltage(t), 10.0)
    t = 5.0
    assert np.isclose(gen.generator_voltage(t), 10.0)


def test_trapezoidal_generator_voltage_falling():
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


def test_trapezoidal_generator_voltage_after_pulse():
    gen = TrapezoidalGenerator(
        U_off=2.0, U_on=8.0, t_rise=1.0, t_on=2.0, t_fall=1.0
    )
    # After pulse ends
    t = 5.0  # t_rise + t_on + t_fall = 4.0
    assert np.isclose(gen.generator_voltage(t), 2.0)


def test_trapezoidal_generator_impedance_constant():
    gen = TrapezoidalGenerator(R_g=15.0)
    freq = np.array([0.0, 1.0, 10.0, 1e6])
    imp = gen.generator_impedance(freq)
    assert np.allclose(imp, 15.0)


def test_gaussian_generator_voltage_peak():
    height = 5.0
    mean = 1e-6
    FWHM = 2e-6
    gen = GaussianGenerator(height=height, mean=mean, FWHM=FWHM)
    # At mean, voltage should be at peak (height)
    assert np.isclose(gen.generator_voltage(mean), height)


def test_gaussian_generator_voltage_far_from_mean():
    gen = GaussianGenerator(height=5.0, mean=0.0, FWHM=1.0)
    # Far from mean, voltage should be close to zero
    assert gen.generator_voltage(100.0) < 1e-6


def test_gaussian_generator_impedance_ideal():
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0)
    freq = np.array([0.0, 1.0, 10.0])
    imp = gen.generator_impedance(freq)
    assert np.allclose(imp, 0.0)


def test_gaussian_generator_impedance_resistive():
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=10.0)
    freq = np.array([0.0, 1.0, 10.0])
    imp = gen.generator_impedance(freq)
    assert np.allclose(imp, 10.0)


def test_gaussian_generator_impedance_resistive_capacitive():
    R_g = 10.0
    C_g = 1e-9
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=R_g, C_g=C_g)
    freq = np.array([0.0, 1e6])
    imp = gen.generator_impedance(freq)
    # At zero frequency, impedance is R_g
    assert np.isclose(imp[0], R_g)
    # At high frequency, impedance should decrease
    assert np.abs(imp[1]) < R_g


def test_gaussian_generator_pure_capacitance():
    # A purely capacitive generator (R_g == 0, C_g != 0) is allowed and
    # behaves as Z_g(f) = 1 / (j 2 pi f C_g), infinite at DC.
    C_g = 1e-9
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=0.0, C_g=C_g)
    freq = np.array([0.0, 1e3, 1e6])
    imp = gen.generator_impedance(freq)
    # Infinite impedance at zero frequency (the capacitor blocks DC).
    assert np.isinf(np.abs(imp[0]))
    # Matches the ideal-capacitor impedance at non-zero frequency.
    expected = 1 / (1j * 2 * np.pi * freq[1:] * C_g)
    assert np.allclose(imp[1:] / expected, 1, rtol=0, atol=1e-12)


def test_gaussian_function_shape():
    # Test the gaussian static method directly
    t = 0.0
    height = 2.0
    mean = 0.0
    FWHM = 1.0
    val = GaussianGenerator.gaussian(t, height, mean, FWHM)
    assert np.isclose(val / height, 1, rtol=0, atol=1e-12)
    # At t = mean + FWHM/2 the value is exactly half the height (by the
    # definition of the full width at half maximum).
    t_half = mean + FWHM / 2
    val_half = GaussianGenerator.gaussian(t_half, height, mean, FWHM)
    assert np.isclose(val_half / (height / 2), 1, rtol=0, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
