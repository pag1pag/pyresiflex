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
    """Minimal concrete ``BaseLoad`` returning a constant impedance.

    Used to exercise the abstract base class without a real load model.
    """

    def __init__(self, purely_resistive: bool, time_varying: bool):
        super().__init__(purely_resistive, time_varying)

    def load_impedance(
        self, t: float = 0.0, frequency: np.ndarray | None = None
    ) -> float:
        # Return a constant value for testing.
        return 42.0


def test_dummy_load_impedance_returns_constant() -> None:
    """Verify ``load_impedance`` returns the constant default value."""
    dummy = DummyLoad(purely_resistive=True, time_varying=False)
    assert dummy.load_impedance() == 42.0


def test_dummy_load_impedance_with_args() -> None:
    """Verify the constant is returned for any time or frequency args."""
    dummy = DummyLoad(purely_resistive=False, time_varying=True)
    assert dummy.load_impedance(t=1.5) == 42.0
    assert dummy.load_impedance(frequency=np.array([1, 2, 3])) == 42.0


def test_base_load_attributes() -> None:
    """Check the constructor stores the resistive/time-varying flags."""
    dummy = DummyLoad(purely_resistive=True, time_varying=True)
    assert dummy.purely_resistive is True
    assert dummy.time_varying is True


#######################################################################
#######################################################################
################ TESTS FOR BASE STEADY IMPEDANCE CLASS ################
#######################################################################
#######################################################################


class DummySteadyImpedance(ComplexImpedanceBaseLoad):
    """Minimal concrete ``ComplexImpedanceBaseLoad`` for testing.

    Validates the frequency array then returns unit impedance, so the
    inherited ``check_frequency`` logic can be exercised directly.
    """

    def __init__(self, purely_resistive: bool):
        super().__init__(purely_resistive)

    def load_impedance(self, frequency: np.ndarray) -> np.ndarray:
        self.check_frequency(frequency)
        return np.ones_like(frequency)


def test_check_frequency_valid() -> None:
    """Ensure a valid 1-D real finite frequency array does not raise."""
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([0.0, 1.0, 2.0])
    # Should not raise.
    dummy.check_frequency(freq)


def test_check_frequency_not_numpy_array() -> None:
    """Check a non-array frequency input raises ``TypeError``."""
    dummy = DummySteadyImpedance(purely_resistive=True)
    with pytest.raises(TypeError, match="Frequency must be a numpy array"):
        # Pass a plain list instead of a numpy array.
        dummy.check_frequency([0.0, 1.0, 2.0])  # type: ignore


def test_check_frequency_not_1d() -> None:
    """Check a 2-D frequency array raises ``ValueError``."""
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="must be a one-dimensional array"):
        dummy.check_frequency(freq)


def test_check_frequency_not_real() -> None:
    """Check a complex-valued frequency array raises ``ValueError``."""
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([1.0, 2.0 + 1j])
    with pytest.raises(ValueError, match="Frequency values must be real"):
        dummy.check_frequency(freq)


def test_check_frequency_not_finite() -> None:
    """Check a frequency array with infinity raises ``ValueError``."""
    dummy = DummySteadyImpedance(purely_resistive=True)
    freq = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="Frequency values must be finite"):
        dummy.check_frequency(freq)


def test_purely_resistive_load_impedance_not_implemented() -> None:
    """Check the abstract resistive ``load_impedance`` raises."""

    class TestLoad(PurelyResistiveBaseLoad):
        def __init__(self) -> None:
            super().__init__(time_varying=False)

        def load_impedance(self, t: float) -> float:
            return super().load_impedance(t)

    # Calling the base implementation raises NotImplementedError.
    with pytest.raises(NotImplementedError, match="Define the load impedance"):
        TestLoad().load_impedance(0.0)


def test_complex_load_impedance_not_implemented() -> None:
    """Check the abstract complex ``load_impedance`` raises.

    The frequency check passes first, then the base implementation
    raises ``NotImplementedError``.
    """

    class TestLoad(ComplexImpedanceBaseLoad):
        def __init__(self) -> None:
            super().__init__(purely_resistive=False)

        def load_impedance(self, frequency: np.ndarray) -> np.ndarray:
            return super().load_impedance(frequency)

    # check_frequency passes, then the base implementation raises.
    with pytest.raises(NotImplementedError, match="Define the load impedance"):
        TestLoad().load_impedance(np.array([1.0, 2.0]))


if __name__ == "__main__":
    pytest.main([__file__])
