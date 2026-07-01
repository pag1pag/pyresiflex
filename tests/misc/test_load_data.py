import pytest

from pyresiflex.misc.load_data import load_minesi_data


def test_load_minesi_data() -> None:
    """Check load_minesi_data returns the expected MINESI parameters."""
    data = load_minesi_data()
    assert data.c == 178160919.5402299
    assert data.Z_c == 68.63171082039064
    assert data.L == 6.2
    assert data.x_meas == 3.1


if __name__ == "__main__":
    pytest.main([__file__])
