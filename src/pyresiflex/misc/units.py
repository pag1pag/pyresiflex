"""Physical constants and unit conversions.

This module provides a class with physical constants and unit conversions.
All units are by default in the International System of Units (SI).

Values are taken from the 2022 CODATA recommended values.
"""

c: float = 299_792_458
r"""Speed of light in vacuum [:math:`\text{m.s}^{-1}`].

See: https://physics.nist.gov/cgi-bin/cuu/Value?c"""

e: float = 1.602_176_634e-19
r"""Elementary charge [:math:`\text{C}`].

See: https://physics.nist.gov/cgi-bin/cuu/Value?e"""

k_b: float = 1.380_649e-23
r"""Boltzmann constant [:math:`\text{J.K}^{-1}`].

See: https://physics.nist.gov/cgi-bin/cuu/Value?k"""

m_e: float = 9.109_383_713_9e-31
r"""Electron mass [:math:`\text{kg}`].

See: https://physics.nist.gov/cgi-bin/cuu/Value?me
"""
