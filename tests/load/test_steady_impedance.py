import numpy as np

from pyresiflex.load.steady_impedance import (
    Capacitance,
    ConstantResistance,
    Inductance,
)


def test_constant_resistance_impedance():
    R = 10.0
    freq = np.array([1, 10, 100])
    cr = ConstantResistance(R)
    impedance = cr.load_impedance(freq)
    assert np.allclose(impedance, R)
    assert impedance.shape == freq.shape


def test_capacitance_impedance():
    C = 1e-6  # 1 uF
    freq = np.array([1, 10, 100])
    cap = Capacitance(C)
    expected = 1 / (1j * 2 * np.pi * freq * C)
    impedance = cap.load_impedance(freq)
    assert np.allclose(impedance, expected)
    assert impedance.shape == freq.shape


def test_capacitance_impedance_zero_frequency():
    C = 1e-6
    freq = np.array([0.0])
    cap = Capacitance(C)
    impedance = cap.load_impedance(freq)
    assert np.isinf(np.abs(impedance[0]))


def test_inductance_impedance():
    L = 1e-3  # 1 mH
    freq = np.array([1, 10, 100])
    ind = Inductance(L)
    expected = 1j * 2 * np.pi * freq * L
    impedance = ind.load_impedance(freq)
    assert np.allclose(impedance, expected)
    assert impedance.shape == freq.shape


def test_inductance_impedance_zero_frequency():
    L = 1e-3
    freq = np.array([0.0])
    ind = Inductance(L)
    impedance = ind.load_impedance(freq)
    assert np.allclose(impedance, 0.0)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
