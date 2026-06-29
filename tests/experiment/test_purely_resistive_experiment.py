import numpy as np
import pytest

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.experiment.purely_resistive_experiment import (
    PurelyResistiveExperiment,
)
from pyresiflex.generator.base_generator import PurelyResistiveBaseGenerator
from pyresiflex.generator.generator_real_impedance import TrapezoidalGenerator
from pyresiflex.load.time_varying_resistance import PlasmaResistanceLinearFall
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

# Bundle returned by the ``reconstruction`` fixture: the experiment under
# test plus everything needed to rebuild and check it (generator, true
# plasma load, cable, time grid and probe position).
Reconstruction = tuple[
    PurelyResistiveExperiment,
    PurelyResistiveBaseGenerator,
    PlasmaResistanceLinearFall,
    PerfectCable,
    np.ndarray,
    float,
]


# ---------------------------------------------------------------------------
# Physics: Ohm's law is recovered only at the load (x_meas = L).
# ---------------------------------------------------------------------------
def test_ohms_law_recovered_at_load() -> None:
    r"""At the load (``x_meas = L``) the propagation delay vanishes.

    With :math:`\tau = (L - x_{meas})/c = 0`, the reconstruction reduces to
    :math:`R_p(t) = Z_c (V_i + V_r)/(V_i - V_r) = V_{meas}(t)/I_{meas}(t)`,
    i.e. Ohm's law, for arbitrary (time-varying) measured signals.
    """
    Z_c, L, c = 50.0, 1.0, 2.0e8
    t_arr = np.linspace(0, 100e-9, 501)
    voltage = 1000.0 + 300.0 * np.sin(2 * np.pi * t_arr / 40e-9)
    current = 5.0 + 2.0 * np.cos(2 * np.pi * t_arr / 55e-9)  # never zero

    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=voltage,
        x_meas_voltage=L,
        experimental_current_time=t_arr,
        experimental_current_value=current,
        x_meas_current=L,
        L=L,
        Z_c=Z_c,
        c=c,
        correct_time_zero=False,
    )

    times = np.linspace(5e-9, 95e-9, 37)
    R_p = expe.compute_plasma_resistance_from_vmeas_and_imeas(
        times, threshold=1e-12
    )
    expected = np.interp(times, t_arr, voltage) / np.interp(
        times, t_arr, current
    )
    assert np.allclose(R_p / expected, 1, rtol=0, atol=1e-12)


def test_ohms_law_not_recovered_away_from_load() -> None:
    """Verify R_p differs from V/I away from the load (``x_meas != L``).

    This confirms the propagation-delay correction actually contributes.
    """
    Z_c, L, c = 50.0, 1.0, 2.0e8
    t_arr = np.linspace(0, 100e-9, 501)
    voltage = 1000.0 + 300.0 * np.sin(2 * np.pi * t_arr / 40e-9)
    current = 5.0 + 2.0 * np.cos(2 * np.pi * t_arr / 55e-9)

    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=voltage,
        x_meas_voltage=L / 2,
        experimental_current_time=t_arr,
        experimental_current_value=current,
        x_meas_current=L / 2,
        L=L,
        Z_c=Z_c,
        c=c,
        correct_time_zero=False,
    )
    times = np.linspace(5e-9, 95e-9, 37)
    R_p = expe.compute_plasma_resistance_from_vmeas_and_imeas(
        times, threshold=1e-12
    )
    expected = np.interp(times, t_arr, voltage) / np.interp(
        times, t_arr, current
    )
    assert not np.allclose(R_p / expected, 1, rtol=0, atol=1e-12)


def test_get_resistance_from_gamma_round_trip() -> None:
    """Check ``Z_l = Z_c (1 + Gamma)/(1 - Gamma)`` inverts Gamma.

    Build Gamma from a known resistance, invert it, and confirm the
    original load resistance is recovered.
    """
    Z_c = 50.0
    R = 137.0
    gamma = (R - Z_c) / (R + Z_c)
    Z_l = PurelyResistiveExperiment.get_resistance_from_gamma(Z_c, gamma)
    assert np.isclose(Z_l / R, 1, rtol=0, atol=1e-12)


def test_time_correction_shifts_signals() -> None:
    """Verify `correct_time_zero` shifts each signal by `x_meas / c`.

    With the flag enabled, ``V_meas``/``I_meas`` at time ``t`` should equal
    the raw signal sampled at ``t - x_meas / c``.
    """
    Z_c, L, c = 50.0, 1.0, 2.0e8
    t_arr = np.linspace(0, 100e-9, 201)
    voltage = np.linspace(0, 2000, 201)
    current = np.linspace(0, 8, 201)
    x_v, x_i = 0.4, 0.6
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=voltage,
        x_meas_voltage=x_v,
        experimental_current_time=t_arr,
        experimental_current_value=current,
        x_meas_current=x_i,
        L=L,
        Z_c=Z_c,
        c=c,
        correct_time_zero=True,
    )
    t = 50e-9
    assert np.isclose(
        expe.V_meas(t) / np.interp(t - x_v / c, t_arr, voltage),
        1,
        rtol=0,
        atol=1e-12,
    )
    assert np.isclose(
        expe.I_meas(t) / np.interp(t - x_i / c, t_arr, current),
        1,
        rtol=0,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Reconstruction from the generator voltage (gamma round-trip).
# ---------------------------------------------------------------------------
@pytest.fixture
def reconstruction(
    perfect_cable: PerfectCable,
    trapezoidal_generator: TrapezoidalGenerator,
) -> Reconstruction:
    """Build synthetic signals from a known linear-fall plasma load.

    Returns the experiment together with the generator, true load, cable,
    time array and probe position used to build it.
    """
    cable = perfect_cable
    generator = trapezoidal_generator
    L, c = cable.L, cable.c
    plasma = PlasmaResistanceLinearFall(
        Z_start=1e3,
        Z_end=10.0,
        t_start_fall=L / c + 5e-9,
        t_end_fall=L / c + 10e-9,
    )
    solution = PurelyResistiveSolution(
        cable=cable, generator=generator, load=plasma
    )
    times = np.arange(0, 30, 0.1) * 1e-9
    x_meas = 2 / 3 * L
    voltage = np.array([solution.V(x_meas, t) for t in times])
    current = np.array([solution.I(x_meas, t) for t in times])
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=times,
        experimental_voltage_value=voltage,
        x_meas_voltage=x_meas,
        experimental_current_time=times,
        experimental_current_value=current,
        x_meas_current=x_meas,
        cable=cable,
        correct_time_zero=False,
    )
    return expe, generator, plasma, cable, times, x_meas


def test_gamma_reconstruction_round_trip(
    reconstruction: Reconstruction,
) -> None:
    """Check the reconstructed load resistance matches the true load.

    Reconstruct Gamma from voltage and from current after the reflected
    wave has returned, convert both to resistances, and require each to
    match the true plasma resistance within a 1e-3 relative tolerance.
    """
    expe, generator, plasma, cable, times, x_meas = reconstruction
    expe.compute_plasma_resistance_from_vmeas_and_vg(
        times, generator=generator, max_n=6
    )
    L, c, Z_c = cable.L, cable.c, cable.Z_c
    # Probe a time after the reflected wave has returned to the probe.
    t_probe = L / c + 8e-9
    gamma_v = expe.get_gamma_load_from_measured_voltage(t_probe, max_n=6)
    gamma_i = expe.get_gamma_load_from_measured_current(t_probe, max_n=6)
    R_v = expe.get_resistance_from_gamma(Z_c, gamma_v)
    R_i = expe.get_resistance_from_gamma(Z_c, gamma_i)
    true_R = plasma.load_impedance(t_probe)
    assert np.isclose(R_v / true_R, 1, rtol=0, atol=1e-3)
    assert np.isclose(R_i / true_R, 1, rtol=0, atol=1e-3)


def test_reconstruct_gamma_x_meas_out_of_range(
    reconstruction: Reconstruction,
) -> None:
    """Ensure a probe position outside the cable raises a ValueError.

    Sets ``x_meas_voltage`` to ``2 * L`` (outside ``[0, L]``) and expects
    the reconstruction to reject it.
    """
    expe, _generator, _plasma, cable, _times, _x_meas = reconstruction
    expe.x_meas_voltage = 2 * cable.L  # outside [0, L]
    with pytest.raises(ValueError, match="x_meas must be between 0 and L"):
        expe.get_gamma_load_from_measured_voltage(10e-9, max_n=5)


def test_compute_from_vg_requires_generator(
    reconstruction: Reconstruction,
) -> None:
    """Ensure passing a non-generator argument raises a TypeError."""
    expe, _generator, _plasma, _cable, times, _x_meas = reconstruction
    bad_generator = "not a generator"
    with pytest.raises(TypeError, match="generator must be an instance"):
        expe.compute_plasma_resistance_from_vmeas_and_vg(
            times,
            generator=bad_generator,  # type: ignore
        )


# ---------------------------------------------------------------------------
# Post-processing of the reconstructed resistance.
# ---------------------------------------------------------------------------
def _masking_experiment() -> tuple[PurelyResistiveExperiment, np.ndarray]:
    """Build an experiment whose reconstruction has a negative window.

    A short window of negative voltage forces negative reconstructed
    resistances (later masked to NaN); the current is constant and
    non-zero so the denominator never vanishes. Returns the experiment
    and the shared time grid.
    """
    Z_c, L, c = 75.0, 1.0, 2.0e8
    t_arr = np.linspace(0, 100e-9, 201)
    voltage = np.full_like(t_arr, 1000.0)
    voltage[120:140] = -500.0  # -> negative reconstructed resistance
    current = np.full_like(t_arr, 4.0)  # constant, non-zero denominator
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=voltage,
        x_meas_voltage=L,
        experimental_current_time=t_arr,
        experimental_current_value=current,
        x_meas_current=L,
        L=L,
        Z_c=Z_c,
        c=c,
        correct_time_zero=False,
    )
    return expe, t_arr


def test_masking_negative_and_channel_formation() -> None:
    """Verify negative R become NaN and pre-formation times become 1 MOhm.

    Without forward-fill interpolation, the NaN samples are dropped from
    the load arrays, so ``times_corrected`` is shorter than the grid.
    """
    expe, t_arr = _masking_experiment()
    cft = 20e-9
    expe.compute_plasma_resistance_from_vmeas_and_imeas(
        t_arr,
        threshold=1e-6,
        channel_formation_time=cft,
        interpolate_with_previous_value=False,
    )
    # Negative-resistance window is NaN in the plotting array.
    assert np.any(np.isnan(expe.Rp_corrected_with_nan[120:140]))
    # Times before channel formation are clamped to 1 MOhm.
    assert np.all(expe.Rp_corrected_with_nan[t_arr < cft] == 1e6)
    # Without interpolation, NaN samples are dropped for the load.
    assert len(expe.times_corrected) < len(t_arr)


def test_masking_interpolate_with_previous_value() -> None:
    """Verify forward-fill keeps the full grid and removes all NaN."""
    expe, t_arr = _masking_experiment()
    expe.compute_plasma_resistance_from_vmeas_and_imeas(
        t_arr,
        threshold=1e-6,
        channel_formation_time=20e-9,
        interpolate_with_previous_value=True,
    )
    assert len(expe.Rp_corrected) == len(t_arr)
    assert not np.any(np.isnan(expe.Rp_corrected))


def test_division_by_zero_warns() -> None:
    """Ensure a zero denominator warns and falls back to 1 MOhm.

    With zero voltage and current, ``V_i = V_r = 0`` makes the
    reconstruction denominator vanish at every sample.
    """
    Z_c, L, c = 75.0, 1.0, 2.0e8
    t_arr = np.linspace(0, 100e-9, 201)
    voltage = np.zeros_like(t_arr)
    current = np.zeros_like(t_arr)  # V_i = V_r = 0 -> denominator zero
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=voltage,
        x_meas_voltage=L,
        experimental_current_time=t_arr,
        experimental_current_value=current,
        x_meas_current=L,
        L=L,
        Z_c=Z_c,
        c=c,
        correct_time_zero=False,
    )
    with pytest.warns(UserWarning, match="denominator are zero"):
        R_p = expe.compute_plasma_resistance_from_vmeas_and_imeas(t_arr)
    assert np.all(R_p == 1e6)


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------
def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a small (time, voltage, current) triplet for constructors.

    The voltage and current are all-ones placeholders; only the array
    shapes matter for the constructor-validation tests.
    """
    t_arr = np.linspace(0, 100e-9, 11)
    return t_arr, np.ones_like(t_arr), np.ones_like(t_arr)


def test_constructor_with_cable() -> None:
    """Verify a passed cable is stored as-is on the experiment."""
    t_arr, v, i = _arrays()
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=v,
        x_meas_voltage=0.5,
        experimental_current_time=t_arr,
        experimental_current_value=i,
        x_meas_current=0.5,
        cable=cable,
    )
    assert expe.cable is cable


def test_constructor_cable_and_parameters_conflict() -> None:
    """Ensure passing both a cable and ``L`` raises a ValueError.

    The cable already carries ``L``/``Z_c``/``c``, so supplying them again
    is ambiguous and rejected.
    """
    t_arr, v, i = _arrays()
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)
    with pytest.raises(ValueError, match="If cable is not None"):
        PurelyResistiveExperiment(
            experimental_voltage_time=t_arr,
            experimental_voltage_value=v,
            x_meas_voltage=0.5,
            experimental_current_time=t_arr,
            experimental_current_value=i,
            x_meas_current=0.5,
            cable=cable,
            L=1.0,
        )


def test_constructor_bad_cable_type() -> None:
    """Ensure a non-cable ``cable`` argument raises a TypeError."""
    t_arr, v, i = _arrays()
    with pytest.raises(TypeError, match="cable must be an instance"):
        PurelyResistiveExperiment(
            experimental_voltage_time=t_arr,
            experimental_voltage_value=v,
            x_meas_voltage=0.5,
            experimental_current_time=t_arr,
            experimental_current_value=i,
            x_meas_current=0.5,
            cable="not a cable",  # type: ignore
        )


def test_constructor_missing_parameters() -> None:
    """Ensure omitting both the cable and L/Z_c/c raises a ValueError."""
    t_arr, v, i = _arrays()
    with pytest.raises(ValueError, match=r"Either `cable`, or all of"):
        PurelyResistiveExperiment(
            experimental_voltage_time=t_arr,
            experimental_voltage_value=v,
            x_meas_voltage=0.5,
            experimental_current_time=t_arr,
            experimental_current_value=i,
            x_meas_current=0.5,
        )


def test_compute_requires_equal_positions() -> None:
    """Ensure reconstruction needs equal voltage/current probe positions.

    With ``x_meas_voltage != x_meas_current`` the V/I reconstruction is
    ill-defined and must raise a ValueError.
    """
    t_arr, v, i = _arrays()
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=v,
        x_meas_voltage=0.4,
        experimental_current_time=t_arr,
        experimental_current_value=i,
        x_meas_current=0.6,
        L=1.0,
        Z_c=50.0,
        c=2.0e8,
    )
    with pytest.raises(ValueError, match=r"`x_meas_voltage` must be equal"):
        expe.compute_plasma_resistance_from_vmeas_and_imeas(t_arr)


# ---------------------------------------------------------------------------
# Plotting (figures are closed by the autouse fixture in conftest).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("correct_time_zero", [False, True])
def test_plot_voltage_and_current(correct_time_zero: bool) -> None:
    """Verify the voltage/current plot builds with and without the shift.

    Parametrized over ``correct_time_zero`` to exercise both the raw and
    time-corrected plotting paths.
    """
    t_arr, v, i = _arrays()
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=v,
        x_meas_voltage=0.4,
        experimental_current_time=t_arr,
        experimental_current_value=i,
        x_meas_current=0.6,
        L=1.0,
        Z_c=50.0,
        c=2.0e8,
        correct_time_zero=correct_time_zero,
    )
    fig, ax_v, ax_i = expe.plot_voltage_and_current()
    assert fig is not None


def test_plot_resistance_before_compute_raises() -> None:
    """Ensure plotting resistance before computing it raises a ValueError."""
    t_arr, v, i = _arrays()
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=t_arr,
        experimental_voltage_value=v,
        x_meas_voltage=0.5,
        experimental_current_time=t_arr,
        experimental_current_value=i,
        x_meas_current=0.5,
        L=1.0,
        Z_c=50.0,
        c=2.0e8,
    )
    with pytest.raises(ValueError, match="resistance has not been computed"):
        expe.plot_resistance(t_arr)


def test_plot_resistance_all_options() -> None:
    """Verify resistance plotting with all options and figure reuse.

    Exercises the whole/corrected/interpolated overlays, then re-plots
    onto the returned ``(fig, ax)`` to confirm they are reused.
    """
    expe, t_arr = _masking_experiment()
    expe.compute_plasma_resistance_from_vmeas_and_imeas(
        t_arr, threshold=1e-6, channel_formation_time=20e-9
    )
    fig, ax = expe.plot_resistance(
        t_arr,
        plot_whole=True,
        plot_corrected=True,
        plot_interpolated=True,
        _also_plot_when_near_cable_impedance=False,
        legend=True,
    )
    assert fig is not None
    # Re-use an existing figure/axes.
    fig2, ax2 = expe.plot_resistance(t_arr, figax=(fig, ax), legend=False)
    assert fig2 is fig


def test_plot_resistance_show(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ``show=True`` calls ``matplotlib.pyplot.show`` exactly once.

    ``pyplot.show`` is monkeypatched to record the call so no window
    opens during the test run.
    """
    expe, t_arr = _masking_experiment()
    expe.compute_plasma_resistance_from_vmeas_and_imeas(
        t_arr, threshold=1e-6, channel_formation_time=20e-9
    )
    called = {}
    monkeypatch.setattr(
        "matplotlib.pyplot.show", lambda *a, **k: called.setdefault("s", True)
    )
    expe.plot_resistance(t_arr, show=True)
    assert called.get("s", False)


if __name__ == "__main__":
    pytest.main([__file__])
