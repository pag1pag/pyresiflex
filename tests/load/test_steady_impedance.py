import numpy as np
import pytest

from pyresiflex.load.steady_impedance import (
    Capacitance,
    ConstantResistance,
    Inductance,
)


def test_constant_resistance_impedance() -> None:
    """Check a resistor impedance equals R at every frequency."""
    R = 10.0
    freq = np.array([1, 10, 100])
    cr = ConstantResistance(R)
    impedance = cr.load_impedance(freq)
    assert np.allclose(impedance, R)
    assert impedance.shape == freq.shape


def test_capacitance_impedance() -> None:
    """Verify capacitor impedance matches 1 / (j 2 pi f C)."""
    C = 1e-6  # 1 uF
    freq = np.array([1, 10, 100])
    cap = Capacitance(C)
    expected = 1 / (1j * 2 * np.pi * freq * C)
    impedance = cap.load_impedance(freq)
    assert np.allclose(impedance, expected)
    assert impedance.shape == freq.shape


def test_capacitance_impedance_zero_frequency() -> None:
    """Check a capacitor impedance is infinite at DC (zero frequency)."""
    C = 1e-6
    freq = np.array([0.0])
    cap = Capacitance(C)
    impedance = cap.load_impedance(freq)
    assert np.isinf(np.abs(impedance[0]))


def test_inductance_impedance() -> None:
    """Verify inductor impedance matches j 2 pi f L."""
    L = 1e-3  # 1 mH
    freq = np.array([1, 10, 100])
    ind = Inductance(L)
    expected = 1j * 2 * np.pi * freq * L
    impedance = ind.load_impedance(freq)
    assert np.allclose(impedance, expected)
    assert impedance.shape == freq.shape


def test_inductance_impedance_zero_frequency() -> None:
    """Check an inductor impedance is zero at DC (zero frequency)."""
    L = 1e-3
    freq = np.array([0.0])
    ind = Inductance(L)
    impedance = ind.load_impedance(freq)
    assert np.allclose(impedance, 0.0)


@pytest.mark.parametrize("C", [0.0, -1e-9])
def test_capacitance_rejects_non_positive(C: float) -> None:
    """Check a zero or negative capacitance is rejected at construction."""
    with pytest.raises(ValueError, match="must be positive"):
        Capacitance(C)


@pytest.mark.parametrize("L", [0.0, -1e-3])
def test_inductance_rejects_non_positive(L: float) -> None:
    """Check a zero or negative inductance is rejected at construction."""
    with pytest.raises(ValueError, match="must be positive"):
        Inductance(L)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
