import pytest

from pyresiflex.cable.cable import PerfectCable


def test_perfect_cable_initialization_valid():
    cable = PerfectCable(L=10.0, Z_c=50.0, c=2.0e8)
    assert cable.L == 10.0
    assert cable.Z_c == 50.0
    assert cable.c == 2.0e8


@pytest.mark.parametrize("L", [0, -1, -100])
def test_perfect_cable_invalid_length(L):
    with pytest.raises(ValueError):
        PerfectCable(L=L, Z_c=50.0, c=2.0e8)


@pytest.mark.parametrize("Z_c", [0, -10, -100])
def test_perfect_cable_invalid_impedance(Z_c):
    with pytest.raises(ValueError):
        PerfectCable(L=10.0, Z_c=Z_c, c=2.0e8)


@pytest.mark.parametrize("c", [0, -1e8, -2.0e8])
def test_perfect_cable_invalid_velocity(c):
    with pytest.raises(ValueError):
        PerfectCable(L=10.0, Z_c=50.0, c=c)


@pytest.mark.parametrize("L", ["ten", None, [], {}])
def test_perfect_cable_type_error_length(L):
    with pytest.raises(TypeError):
        PerfectCable(L=L, Z_c=50.0, c=2.0e8)


@pytest.mark.parametrize("Z_c", ["fifty", None, [], {}])
def test_perfect_cable_type_error_impedance(Z_c):
    with pytest.raises(TypeError):
        PerfectCable(L=10.0, Z_c=Z_c, c=2.0e8)


@pytest.mark.parametrize("c", ["fast", None, [], {}])
def test_perfect_cable_type_error_velocity(c):
    with pytest.raises(TypeError):
        PerfectCable(L=10.0, Z_c=50.0, c=c)


@pytest.mark.parametrize(
    "L, Z_c, c, exc_type, msg",
    [
        (
            "abc",
            50.0,
            2.0e8,
            TypeError,
            "Length of the transmission line `L` must be a number.",
        ),
        (
            10.0,
            "xyz",
            2.0e8,
            TypeError,
            "Characteristic impedance `Z_c` must be a number.",
        ),
        (
            10.0,
            50.0,
            "speed",
            TypeError,
            "Wave velocity `c` must be a number.",
        ),
        (
            0,
            50.0,
            2.0e8,
            ValueError,
            "Length of the transmission line `L` must be positive.",
        ),
        (
            10.0,
            0,
            2.0e8,
            ValueError,
            "Characteristic impedance `Z_c` must be positive.",
        ),
        (10.0, 50.0, 0, ValueError, "Wave velocity `c` must be positive."),
        (
            10.0,
            50.0,
            3.1e8,
            ValueError,
            "Wave velocity `c` cannot exceed the speed of light.",
        ),
    ],
)
def test_check_parameters_exceptions(L, Z_c, c, exc_type, msg):
    with pytest.raises(exc_type) as excinfo:
        PerfectCable.check_parameters(L, Z_c, c)
    assert msg in str(excinfo.value)


def test_check_parameters_valid():
    # Should not raise any exception
    PerfectCable.check_parameters(10.0, 50.0, 2.0e8)


if __name__ == "__main__":
    pytest.main([__file__])
