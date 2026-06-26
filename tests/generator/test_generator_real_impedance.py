import numpy as np
import pytest

from pyresiflex.generator.generator_real_impedance import (
    FromMeasurementGenerator,
    GaussianGenerator,
    TrapezoidalGenerator,
)


def test_trapezoidal_generator_voltage():
    gen = TrapezoidalGenerator(
        R_g=2.0, U_off=1.0, U_on=5.0, t_rise=2.0, t_on=3.0, t_fall=2.0
    )
    # Before pulse
    assert gen.generator_voltage(-1.0) == 1.0
    # Start of rise
    assert gen.generator_voltage(0.0) == 1.0
    # During rise
    assert gen.generator_voltage(1.0) == 1.0 + (5.0 - 1.0) * 1.0 / 2.0
    # End of rise
    assert gen.generator_voltage(2.0) == 5.0
    # Plateau
    assert gen.generator_voltage(3.0) == 5.0
    assert gen.generator_voltage(4.0) == 5.0
    # Start of fall
    assert gen.generator_voltage(5.0) == 5.0
    # During fall
    assert (
        gen.generator_voltage(6.0)
        == 5.0 - (5.0 - 1.0) * (6.0 - 2.0 - 3.0) / 2.0
    )
    # After pulse
    assert gen.generator_voltage(7.0) == 1.0


def test_trapezoidal_generator_impedance():
    gen = TrapezoidalGenerator(R_g=3.3)
    assert gen.generator_impedance() == 3.3


def test_gaussian_generator_voltage():
    gen = GaussianGenerator(height=10.0, mean=5.0, FWHM=2.0, R_g=1.0)
    # At mean
    v_mean = gen.generator_voltage(5.0)
    assert np.isclose(v_mean, 10.0)
    # Far from mean
    v_far = gen.generator_voltage(0.0)
    assert v_far < 1.0
    # Symmetry
    assert np.isclose(
        gen.generator_voltage(5.0 + 1.0), gen.generator_voltage(5.0 - 1.0)
    )


def test_gaussian_generator_impedance():
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=2.2)
    assert gen.generator_impedance() == 2.2


def test_from_measurement_generator_voltage():
    def v_meas(t):
        return 2.0 * t + 1.0

    gen = FromMeasurementGenerator(R_g=5.0, V_meas=v_meas)
    assert gen.generator_voltage(0.0) == 1.0
    assert gen.generator_voltage(2.0) == 5.0
    assert gen.generator_voltage(-1.0) == -1.0


def test_from_measurement_generator_impedance():
    gen = FromMeasurementGenerator(R_g=7.7, V_meas=lambda t: t)
    assert gen.generator_impedance() == 7.7


if __name__ == "__main__":
    pytest.main([__file__])
