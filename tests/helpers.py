"""Shared assertion helpers for the test suite."""

import numpy as np


def relative_close(
    actual: float | complex | np.ndarray,
    expected: float | complex | np.ndarray,
    atol: float,
) -> bool:
    """Return whether ``actual / expected`` equals 1 within ``atol``.

    Pure relative comparison (``rtol = 0``) that works for scalars and
    arrays alike. The reference ``expected`` must be non-zero.

    Parameters
    ----------
    actual : float or complex or numpy.ndarray
        Value(s) under test.
    expected : float or complex or numpy.ndarray
        Non-zero reference value(s).
    atol : float
        Absolute tolerance on the relative error ``|actual / expected - 1|``.

    Returns
    -------
    bool
        True if every element agrees to within ``atol``.
    """
    ratio = np.asarray(actual) / np.asarray(expected)
    return bool(np.all(np.isclose(ratio, 1.0, rtol=0.0, atol=atol)))
