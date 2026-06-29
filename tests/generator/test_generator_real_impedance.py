import numpy as np
import pytest

from pyresiflex.generator.generator_real_impedance import (
    ConstantGenerator,
    FromMeasurementGenerator,
    GaussianGenerator,
    TrapezoidalGenerator,
)


def test_trapezoidal_generator_voltage() -> None:
    """Verify the trapezoidal voltage across every pulse segment.

    Check the off level, the linear rise, the plateau, the linear
    fall, and the return to the off level after the pulse.
    """
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


def test_trapezoidal_generator_impedance() -> None:
    """Check the trapezoidal impedance equals the constant ``R_g``."""
    gen = TrapezoidalGenerator(R_g=3.3)
    assert gen.generator_impedance() == 3.3


def test_gaussian_generator_voltage() -> None:
    """Verify the gaussian voltage peak, decay, and symmetry.

    Check the peak equals ``height`` at the mean, the value is small
    far from the mean, and the shape is symmetric about the mean.
    """
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


def test_gaussian_generator_impedance() -> None:
    """Check the gaussian impedance equals the constant ``R_g``."""
    gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1.0, R_g=2.2)
    assert gen.generator_impedance() == 2.2


def test_from_measurement_generator_voltage() -> None:
    """Verify the voltage delegates to the supplied measurement func.

    Sample the affine measurement ``2*t + 1`` at three times to confirm
    the generator returns the callable's value unchanged.
    """

    def v_meas(t: float) -> float:
        return 2.0 * t + 1.0

    gen = FromMeasurementGenerator(R_g=5.0, V_meas=v_meas)
    assert gen.generator_voltage(0.0) == 1.0
    assert gen.generator_voltage(2.0) == 5.0
    assert gen.generator_voltage(-1.0) == -1.0


def test_from_measurement_generator_impedance() -> None:
    """Check the measurement generator impedance equals ``R_g``."""
    gen = FromMeasurementGenerator(R_g=7.7, V_meas=lambda t: t)
    assert gen.generator_impedance() == 7.7


def test_constant_generator_voltage() -> None:
    """Verify the constant voltage is zero before ``t = 0``, ``U_g`` after.

    Check the step behaviour: no voltage for negative time and the
    constant level ``U_g`` from ``t = 0`` onward.
    """
    gen = ConstantGenerator(R_g=3.0, U_g=12.0)
    # No voltage before t = 0.
    assert gen.generator_voltage(-1.0) == 0.0
    # Constant voltage afterwards.
    assert gen.generator_voltage(0.0) == 12.0
    assert gen.generator_voltage(5.0) == 12.0


def test_constant_generator_impedance() -> None:
    """Check the constant generator impedance equals ``R_g``."""
    gen = ConstantGenerator(R_g=4.5, U_g=1.0)
    assert gen.generator_impedance() == 4.5


if __name__ == "__main__":
    pytest.main([__file__])
