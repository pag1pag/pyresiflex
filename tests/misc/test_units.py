import pytest

from pyresiflex.misc import units


def test_physical_constants_values():
    # 2022 CODATA recommended values (exact for c_0, e, k_b).
    assert units.c_0 == 299_792_458
    assert units.e == pytest.approx(1.602_176_634e-19)
    assert units.k_b == pytest.approx(1.380_649e-23)
    assert units.m_e == pytest.approx(9.109_383_713_9e-31)


if __name__ == "__main__":
    pytest.main([__file__])
