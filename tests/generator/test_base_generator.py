import numpy as np
import pytest

from pyresiflex.generator.base_generator import (
    BaseGenerator,
    ComplexImpedanceBaseGenerator,
)


class DummyComplexImpedanceGenerator(ComplexImpedanceBaseGenerator):
    def generator_impedance(self, frequency: np.ndarray) -> np.ndarray:
        self.check_frequency(frequency)
        # Dummy implementation: return frequency as impedance
        return frequency

    def generator_voltage(self, t: float) -> float:
        return 1.0


def test_check_frequency_valid():
    freq = np.array([0.0, 1.0, 10.0])
    # Should not raise
    ComplexImpedanceBaseGenerator.check_frequency(freq)


@pytest.mark.parametrize(
    "freq",
    [
        [0.0, 1.0, 2.0],  # not np.ndarray
        np.array([[1.0, 2.0], [3.0, 4.0]]),  # not 1D
        np.array([1.0, 2.0, 3.0j]),  # complex values
        np.array([1.0, np.inf, 3.0]),  # non-finite
    ],
)
def test_check_frequency_invalid(freq):
    if not isinstance(freq, np.ndarray):
        with pytest.raises(TypeError):
            ComplexImpedanceBaseGenerator.check_frequency(freq)
    else:
        with pytest.raises(ValueError):
            ComplexImpedanceBaseGenerator.check_frequency(freq)


def test_generator_impedance_returns_expected():
    dummy = DummyComplexImpedanceGenerator()
    freq = np.array([1.0, 2.0, 3.0])
    result = dummy.generator_impedance(freq)
    assert np.array_equal(result, freq)


def test_abstract_generator_impedance_raises():
    class TestGen(ComplexImpedanceBaseGenerator):
        def generator_voltage(self, t: float) -> float:
            return 1.0

    with pytest.raises(TypeError):
        TestGen()


def test_base_generator_voltage_not_implemented():
    class TestGen(BaseGenerator):
        def __init__(self):
            super().__init__(purely_resistive=True)

        def generator_voltage(self, t: float) -> float:
            return super().generator_voltage(t)

    with pytest.raises(NotImplementedError):
        TestGen().generator_voltage(0.0)


def test_complex_generator_impedance_not_implemented():
    class TestGen(ComplexImpedanceBaseGenerator):
        def generator_voltage(self, t: float) -> float:
            return 1.0

        def generator_impedance(self, frequency: np.ndarray) -> np.ndarray:
            return super().generator_impedance(frequency)

    # check_frequency passes, then the base implementation raises.
    with pytest.raises(NotImplementedError):
        TestGen().generator_impedance(np.array([1.0, 2.0]))


if __name__ == "__main__":
    pytest.main([__file__])
