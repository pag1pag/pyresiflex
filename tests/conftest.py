"""Shared fixtures and configuration for the test suite."""

from collections.abc import Iterator

import matplotlib
import matplotlib.pyplot as plt
import pytest

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.generator_real_impedance import TrapezoidalGenerator

# Use a non-interactive backend so tests never open windows. Set here once,
# before any test module imports pyplot, instead of per file.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close every matplotlib figure after each test.

    Figures created through pyplot are held for the whole session, so they
    accumulate across tests and eventually trip matplotlib's "more than 20
    figures" warning -- which is fatal when warnings are errors. Closing
    them after each test keeps the suite isolated and warning-free.
    """
    yield
    plt.close("all")


@pytest.fixture
def perfect_cable() -> PerfectCable:
    """Return a short perfect cable shared across tests."""
    return PerfectCable(L=0.3, Z_c=75.0, c=1.66e8)


@pytest.fixture
def trapezoidal_generator() -> TrapezoidalGenerator:
    """Return a 10 kV trapezoidal generator (1 Ohm source)."""
    return TrapezoidalGenerator(R_g=1.0, U_on=10e3)
