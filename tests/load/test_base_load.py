import numpy as np
import pytest

from pyresiflex.load.base_load import BaseLoad, BaseSteadyImpedance


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


def test_base_load_instantiation():
    # Should not be able to instantiate BaseLoad directly
    with pytest.raises(TypeError):
        BaseLoad(purely_resistive=True, time_varying=False)  # type: ignore


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


def test_base_load_abstract_method_raises():
    class IncompleteLoad(BaseLoad):
        pass

    with pytest.raises(TypeError):
        IncompleteLoad(purely_resistive=True, time_varying=False)  # type: ignore


#######################################################################
#######################################################################
################ TESTS FOR BASE STEADY IMPEDANCE CLASS ################
#######################################################################
#######################################################################


class DummySteadyImpedance(BaseSteadyImpedance):
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
    with pytest.raises(TypeError):
        # Not a numpy array
        dummy.check_frequency([0.0, 1.0, 2.0])  # type: ignore


def test_check_frequency_not_1d():
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        dummy.check_frequency(freq)


def test_check_frequency_not_real():
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([1.0, 2.0 + 1j])
    with pytest.raises(ValueError):
        dummy.check_frequency(freq)


if __name__ == "__main__":
    pytest.main([__file__])
