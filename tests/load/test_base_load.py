import numpy as np
import pytest

from pyresiflex.load.base_load import (
    BaseLoad,
    ComplexImpedanceBaseLoad,
    PurelyResistiveBaseLoad,
)


#######################################################################
#######################################################################
###################### TESTS FOR BASE LOAD CLASS ######################
#######################################################################
#######################################################################
class DummyLoad(BaseLoad):
    def __init__(self, purely_resistive, time_varying):
        super().__init__(purely_resistive, time_varying)

    def load_impedance(self, t=0.0, frequency=None):
        # Return a constant value for testing
        return 42.0


def test_dummy_load_impedance_returns_constant():
    dummy = DummyLoad(purely_resistive=True, time_varying=False)
    assert dummy.load_impedance() == 42.0


def test_dummy_load_impedance_with_args():
    dummy = DummyLoad(purely_resistive=False, time_varying=True)
    assert dummy.load_impedance(t=1.5) == 42.0
    assert dummy.load_impedance(frequency=np.array([1, 2, 3])) == 42.0


def test_base_load_attributes():
    dummy = DummyLoad(purely_resistive=True, time_varying=True)
    assert dummy.purely_resistive is True
    assert dummy.time_varying is True


#######################################################################
#######################################################################
################ TESTS FOR BASE STEADY IMPEDANCE CLASS ################
#######################################################################
#######################################################################


class DummySteadyImpedance(ComplexImpedanceBaseLoad):
    def __init__(self, purely_resistive):
        super().__init__(purely_resistive)

    def load_impedance(self, frequency: np.ndarray):
        self.check_frequency(frequency)
        return np.ones_like(frequency)


def test_check_frequency_valid():
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([0.0, 1.0, 2.0])
    # Should not raise
    dummy.check_frequency(freq)


def test_check_frequency_not_numpy_array():
    dummy = DummySteadyImpedance(purely_resistive=True)
    with pytest.raises(TypeError, match="Frequency must be a numpy array"):
        # Not a numpy array
        dummy.check_frequency([0.0, 1.0, 2.0])  # type: ignore


def test_check_frequency_not_1d():
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="must be a one-dimensional array"):
        dummy.check_frequency(freq)


def test_check_frequency_not_real():
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([1.0, 2.0 + 1j])
    with pytest.raises(ValueError, match="Frequency values must be real"):
        dummy.check_frequency(freq)


def test_check_frequency_not_finite():
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="Frequency values must be finite"):
        dummy.check_frequency(freq)


def test_purely_resistive_load_impedance_not_implemented():
    class TestLoad(PurelyResistiveBaseLoad):
        def __init__(self):
            super().__init__(time_varying=False)

        def load_impedance(self, t: float) -> float:
            return super().load_impedance(t)

    with pytest.raises(NotImplementedError, match="Define the load impedance"):
        TestLoad().load_impedance(0.0)


def test_complex_load_impedance_not_implemented():
    class TestLoad(ComplexImpedanceBaseLoad):
        def __init__(self):
            super().__init__(purely_resistive=False)

        def load_impedance(self, frequency: np.ndarray) -> np.ndarray:
            return super().load_impedance(frequency)

    # Check_frequency passes, then the base implementation raises.
    with pytest.raises(NotImplementedError, match="Define the load impedance"):
        TestLoad().load_impedance(np.array([1.0, 2.0]))


if __name__ == "__main__":
    pytest.main([__file__])
