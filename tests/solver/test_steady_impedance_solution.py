from math import erfc

import numpy as np
import pytest

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.generator_complex_impedance import (
    GaussianGenerator,
)
from pyresiflex.generator.generator_real_impedance import (
    TrapezoidalGenerator as RealTrapezoidalGenerator,
)
from pyresiflex.load.steady_impedance import Capacitance, ConstantResistance
from pyresiflex.load.time_varying_resistance import (
    ConstantResistance as ResistiveTimeLoad,
)
from pyresiflex.misc.utils import gaussian_fwhm_from_sigma
from pyresiflex.solver.steady_impedance_solution import (
    SteadyImpedanceSolution,
)
from tests.helpers import relative_close

# Solution built by the ``matched_gaussian_capacitor`` fixture, paired with
# the physical parameters needed to evaluate the analytical solution.
MatchedCapacitor = tuple[SteadyImpedanceSolution, dict[str, float]]


@pytest.fixture(scope="module")
def matched_gaussian_capacitor() -> MatchedCapacitor:
    """Build a matched-generator / capacitor-load configuration.

    Module-scoped: the (expensive) 8000-point FFT solution is only read by
    the tests, so it is built once and shared. Returns the solution together
    with the physical parameters needed to evaluate the analytical solution
    at the load.
    """
    Z_c, c, L = 50.0, 2.0e8, 1.0
    V0 = 1.0
    sigma = 3e-9
    a = 1.0 / sigma**2
    t0 = 50e-9  # Center the Gaussian inside the FFT window.
    FWHM = gaussian_fwhm_from_sigma(sigma)
    C = 1e-10
    tau = Z_c * C  # = 5 ns

    # Matched generator: Z_g = R_g = Z_c (so Gamma_g = 0), Gaussian voltage.
    generator = GaussianGenerator(
        height=V0, mean=t0, FWHM=FWHM, R_g=Z_c, C_g=0.0
    )
    load = Capacitance(C=C)
    cable = PerfectCable(L=L, Z_c=Z_c, c=c)
    solution = SteadyImpedanceSolution(
        cable, generator, load, nb_points_ft=8000, max_time_ft=200e-9
    )
    return solution, dict(Z_c=Z_c, c=c, L=L, V0=V0, a=a, t0=t0, tau=tau)


def test_capacitor_load_matches_analytic_solution(
    matched_gaussian_capacitor: MatchedCapacitor,
) -> None:
    r"""Check the capacitor load voltage matches its analytic solution.

    With a matched generator (:math:`Z_g = Z_c`, hence :math:`\Gamma_g = 0`)
    and a Gaussian generator voltage :math:`V_g(t) = V_0 e^{-a (t-t_0)^2}`,
    the voltage at a capacitive load obeys the first-order ODE

    .. math::
        \tau \frac{dV_L}{dt} + V_L = V_g(t - L/c), \quad \tau = Z_c C,

    whose solution is

    .. math::
        V\!\left(L, t + L/c\right) = \sqrt{\tfrac{\pi}{a}} \tfrac{V_0}{2\tau}
        \exp\!\left(\tfrac{1}{4 a \tau^2} - \tfrac{t}{\tau}\right)
        \mathrm{erfc}\!\left(\tfrac{1}{2 \tau \sqrt{a}} - \sqrt{a}\, t\right).
    """
    solution, p = matched_gaussian_capacitor
    c, L = p["c"], p["L"]
    V0, a, t0, tau = p["V0"], p["a"], p["t0"], p["tau"]

    def analytic_at_load(t_lab: float) -> float:
        """Return the closed-form load voltage at lab time ``t_lab``."""
        # t_lab is the absolute (lab) time at the load x = L.
        s = t_lab - L / c - t0
        prefactor = np.sqrt(np.pi / a) * V0 / (2 * tau)
        return (
            prefactor
            * np.exp(1 / (4 * a * tau**2) - s / tau)
            * erfc(1 / (2 * tau * np.sqrt(a)) - np.sqrt(a) * s)
        )

    t_lab = np.linspace(48e-9, 100e-9, 21) + L / c
    numerical = np.array([solution.V(L, t) for t in t_lab])
    analytic = np.array([analytic_at_load(t) for t in t_lab])

    # No NaN must remain (the DC bin Z_l(0) = inf is handled as Gamma_l -> 1).
    assert not np.any(np.isnan(numerical))
    rel_l2 = np.linalg.norm(numerical - analytic) / np.linalg.norm(analytic)
    assert rel_l2 < 1e-11

    # Pointwise relative agreement with the analytical solution.
    assert relative_close(numerical, analytic, 1e-11)


def test_capacitor_gamma_l_dc_is_open_circuit(
    matched_gaussian_capacitor: MatchedCapacitor,
) -> None:
    """Check the capacitor load reflection at DC tends to an open circuit.

    At DC the capacitor is an open circuit, so the load reflection
    coefficient ``gamma_l(0)`` must equal 1 and stay finite everywhere.
    """
    solution, _ = matched_gaussian_capacitor
    gamma_l = solution.gamma_l(solution.f)
    # f[0] is the DC component for numpy.fft.fftfreq.
    assert relative_close(gamma_l[0], 1.0, 1e-12)
    assert np.all(np.isfinite(gamma_l))


def test_per_generation_waves_sum_to_total() -> None:
    """Check partial sums of per-generation waves match the closed form.

    ``V_incident_total`` is the closed-form geometric-series sum of the
    per-generation incident waves ``V_incident(t, n)`` (and likewise for
    the reflected waves). A mismatched, purely resistive setup makes
    several generations contribute, so the partial sums must converge to
    the closed-form totals.
    """
    # Mismatched resistive generator and load -> several generations.
    generator = GaussianGenerator(
        height=1.0, mean=30e-9, FWHM=8e-9, R_g=10.0, C_g=0.0
    )
    load = ConstantResistance(R=150.0)
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)
    solution = SteadyImpedanceSolution(
        cable, generator, load, nb_points_ft=4000, max_time_ft=200e-9
    )

    t = 100e-9
    n_max = 25
    incident_total = solution.V_incident_total(t)
    reflected_total = solution.V_reflected_total(t)
    incident_partial = sum(solution.V_incident(t, n) for n in range(n_max + 1))
    reflected_partial = sum(
        solution.V_reflected(t, n) for n in range(n_max + 1)
    )

    assert relative_close(incident_partial, incident_total, 1e-9)
    assert relative_close(reflected_partial, reflected_total, 1e-9)


def test_pure_capacitive_generator_is_finite() -> None:
    """Check a purely capacitive generator yields a finite solution.

    ``Z_g(0) = inf`` blocks the DC component (``alpha_g -> 0``) and fully
    reflects it (``gamma_g -> 1``); the solution must remain NaN-free.
    """
    generator = GaussianGenerator(
        height=1.0, mean=30e-9, FWHM=8e-9, R_g=0.0, C_g=1e-10
    )
    load = ConstantResistance(R=150.0)
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)
    solution = SteadyImpedanceSolution(
        cable, generator, load, nb_points_ft=4000, max_time_ft=200e-9
    )
    # DC component is blocked by the series capacitor and fully reflected.
    # (alpha_g -> 0 is a comparison against zero, so it stays absolute.)
    assert np.isclose(solution.alpha_g(solution.f)[0], 0.0, rtol=0, atol=1e-12)
    assert relative_close(solution.gamma_g(solution.f)[0], 1.0, 1e-12)
    t = np.linspace(20e-9, 120e-9, 11)
    voltage = np.array([solution.V(1.0, ti) for ti in t])
    assert np.all(np.isfinite(voltage))


def test_capacitive_generator_and_capacitive_load_is_finite() -> None:
    """Check a capacitive generator AND capacitive load stay NaN-free.

    Both ends are open at DC (ω=0), so ``gamma_g = gamma_l = 1`` and the
    geometric-series denominator ``1 - gamma_g * gamma_l`` vanishes at the
    DC bin. The numerator vanishes too (``alpha_g = 0``), so the 0/0 term
    must resolve to 0 rather than poisoning the sum with NaN. Regression
    guard for the unguarded ``V_incident_total``/``V_reflected_total``
    denominators (also a hard error under ``filterwarnings = ['error']``).
    """
    generator = GaussianGenerator(
        height=1.0, mean=30e-9, FWHM=8e-9, R_g=0.0, C_g=1e-10
    )
    load = Capacitance(C=1e-10)
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)
    solution = SteadyImpedanceSolution(
        cable, generator, load, nb_points_ft=2000, max_time_ft=200e-9
    )
    # The closed-form total waves must be finite at the DC bin...
    assert np.isfinite(solution.V_incident_total(30e-9))
    assert np.isfinite(solution.V_reflected_total(30e-9))
    # ... and so must a full solve (voltage, current and energy).
    solution.solve(np.linspace(0, cable.L, 4), np.linspace(20e-9, 120e-9, 11))
    assert np.all(np.isfinite(solution.voltage))
    assert np.all(np.isfinite(solution.current))
    assert np.all(np.isfinite(solution.energy))


def test_shannon_nyquist_warning() -> None:
    """Check a too-large FFT time step triggers a Shannon-Nyquist warning.

    With ``dt = max_time_ft / nb_points_ft`` above 1 ns, the constructor
    must emit the under-sampling ``UserWarning``.
    """
    generator = GaussianGenerator(
        height=1.0, mean=30e-9, FWHM=8e-9, R_g=10.0, C_g=0.0
    )
    load = ConstantResistance(R=150.0)
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)
    with pytest.warns(UserWarning, match="too large for Fourier Transform"):
        # dt = 1e-6 / 10 = 1e-7 s > 1 ns.
        SteadyImpedanceSolution(
            cable, generator, load, nb_points_ft=10, max_time_ft=1e-6
        )


def test_invalid_constructor_arguments() -> None:
    """Check the constructor validates cable/generator/load types.

    A non-cable object, a purely resistive (non-complex-impedance)
    generator, and a purely resistive load must each raise ``TypeError``
    naming the offending argument.
    """
    generator = GaussianGenerator(
        height=1.0, mean=30e-9, FWHM=8e-9, R_g=10.0, C_g=0.0
    )
    load = ConstantResistance(R=150.0)
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)

    bad_cable = "not a cable"
    # A purely resistive generator is not a complex-impedance generator.
    bad_generator = RealTrapezoidalGenerator(R_g=10.0)
    # A purely resistive load is not a complex-impedance load.
    bad_load = ResistiveTimeLoad(R=150.0)

    with pytest.raises(TypeError, match=r"`cable` must be an instance"):
        SteadyImpedanceSolution(bad_cable, generator, load)  # type: ignore
    with pytest.raises(TypeError, match=r"`generator` must be an instance"):
        SteadyImpedanceSolution(cable, bad_generator, load)  # type: ignore
    with pytest.raises(TypeError, match=r"`load` must be an instance"):
        SteadyImpedanceSolution(cable, generator, bad_load)  # type: ignore


if __name__ == "__main__":
    pytest.main([__file__])
