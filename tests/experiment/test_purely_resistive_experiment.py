import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.experiment.purely_resistive_experiment import (
    PurelyResistiveExperiment,
)
from pyresiflex.generator.generator_real_impedance import TrapezoidalGenerator
from pyresiflex.load.time_varying_resistance import PlasmaResistanceLinearFall
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

matplotlib.use("Agg")  # Non-interactive backend for testing.


# ---------------------------------------------------------------------------
# Physics: Ohm's law is recovered only at the load (x_meas = L).
# ---------------------------------------------------------------------------
def test_ohms_law_recovered_at_load():
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
    assert np.allclose(R_p, expected)


def test_ohms_law_not_recovered_away_from_load():
    """Away from the load (``x_meas != L``) R_p is not simply V/I.

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
    assert not np.allclose(R_p, expected)


def test_get_resistance_from_gamma_round_trip():
    """``Z_l = Z_c (1 + Gamma)/(1 - Gamma)`` inverts the reflection coeff."""
    Z_c = 50.0
    R = 137.0
    gamma = (R - Z_c) / (R + Z_c)
    Z_l = PurelyResistiveExperiment.get_resistance_from_gamma(Z_c, gamma)
    assert np.isclose(Z_l, R)


def test_time_correction_shifts_signals():
    """`correct_time_zero` shifts the measured signals by `x_meas / c`."""
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
    assert np.isclose(expe.V_meas(t), np.interp(t - x_v / c, t_arr, voltage))
    assert np.isclose(expe.I_meas(t), np.interp(t - x_i / c, t_arr, current))


# ---------------------------------------------------------------------------
# Reconstruction from the generator voltage (gamma round-trip).
# ---------------------------------------------------------------------------
def _reconstruction_setup():
    L, c, Z_c = 0.3, 1.66e8, 75.0
    cable = PerfectCable(L=L, Z_c=Z_c, c=c)
    generator = TrapezoidalGenerator(R_g=1.0, U_on=10e3)
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


def test_gamma_reconstruction_round_trip():
    """The reconstructed load resistance matches the true plasma load."""
    expe, generator, plasma, cable, times, x_meas = _reconstruction_setup()
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
    assert np.isclose(R_v, true_R, rtol=5e-3)
    assert np.isclose(R_i, true_R, rtol=5e-3)


def test_reconstruct_gamma_x_meas_out_of_range():
    """A probe position outside the cable raises a ValueError."""
    expe, generator, plasma, cable, times, x_meas = _reconstruction_setup()
    expe.x_meas_voltage = 2 * cable.L  # outside [0, L]
    with pytest.raises(ValueError):
        expe.get_gamma_load_from_measured_voltage(10e-9, max_n=5)


def test_compute_from_vg_requires_generator():
    """A non-generator argument raises a TypeError."""
    expe, generator, plasma, cable, times, x_meas = _reconstruction_setup()
    bad_generator = "not a generator"
    with pytest.raises(TypeError):
        expe.compute_plasma_resistance_from_vmeas_and_vg(
            times,
            generator=bad_generator,
        )


# ---------------------------------------------------------------------------
# Post-processing of the reconstructed resistance.
# ---------------------------------------------------------------------------
def _masking_experiment():
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


def test_masking_negative_and_channel_formation():
    """Negative resistances become NaN; early times become 1 MOhm."""
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


def test_masking_interpolate_with_previous_value():
    """Forward-filling keeps the full time grid and removes all NaN."""
    expe, t_arr = _masking_experiment()
    expe.compute_plasma_resistance_from_vmeas_and_imeas(
        t_arr,
        threshold=1e-6,
        channel_formation_time=20e-9,
        interpolate_with_previous_value=True,
    )
    assert len(expe.Rp_corrected) == len(t_arr)
    assert not np.any(np.isnan(expe.Rp_corrected))


def test_division_by_zero_warns():
    """A zero denominator triggers a warning and a 1 MOhm fallback."""
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
    with pytest.warns(UserWarning):
        R_p = expe.compute_plasma_resistance_from_vmeas_and_imeas(t_arr)
    assert np.all(R_p == 1e6)


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------
def _arrays():
    t_arr = np.linspace(0, 100e-9, 11)
    return t_arr, np.ones_like(t_arr), np.ones_like(t_arr)


def test_constructor_with_cable():
    t_arr, v, i = _arrays()
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)
    expe = PurelyResistiveExperiment(t_arr, v, 0.5, t_arr, i, 0.5, cable=cable)
    assert expe.cable is cable


def test_constructor_cable_and_parameters_conflict():
    t_arr, v, i = _arrays()
    cable = PerfectCable(L=1.0, Z_c=50.0, c=2.0e8)
    with pytest.raises(ValueError):
        # Conflict: L is provided both in the cable and as parameters.
        PurelyResistiveExperiment(
            t_arr, v, 0.5, t_arr, i, 0.5, cable=cable, L=1.0
        )


def test_constructor_bad_cable_type():
    t_arr, v, i = _arrays()
    with pytest.raises(TypeError):
        PurelyResistiveExperiment(
            t_arr,
            v,
            0.5,
            t_arr,
            i,
            0.5,
            cable="not a cable",  # type: ignore
        )


def test_constructor_missing_parameters():
    t_arr, v, i = _arrays()
    with pytest.raises(ValueError):
        # Either a cable or the parameters L, Z_c, c must be provided.
        PurelyResistiveExperiment(t_arr, v, 0.5, t_arr, i, 0.5)


def test_compute_requires_equal_positions():
    t_arr, v, i = _arrays()
    expe = PurelyResistiveExperiment(
        t_arr, v, 0.4, t_arr, i, 0.6, L=1.0, Z_c=50.0, c=2.0e8
    )
    with pytest.raises(ValueError):
        # The voltage and current probe positions must be equal for
        # the reconstruction with V_meas and I_meas.
        expe.compute_plasma_resistance_from_vmeas_and_imeas(t_arr)


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("correct_time_zero", [False, True])
def test_plot_voltage_and_current(correct_time_zero):
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
    plt.close(fig)


def test_plot_resistance_before_compute_raises():
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
    with pytest.raises(ValueError):
        # `compute_plasma_resistance_from_vmeas_and_imeas` must be called
        # before plotting the resistance.
        expe.plot_resistance(t_arr)


def test_plot_resistance_all_options():
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
    plt.close(fig)


def test_plot_resistance_show(monkeypatch: pytest.MonkeyPatch):
    expe, t_arr = _masking_experiment()
    expe.compute_plasma_resistance_from_vmeas_and_imeas(
        t_arr, threshold=1e-6, channel_formation_time=20e-9
    )
    called = {}
    monkeypatch.setattr(
        "matplotlib.pyplot.show", lambda *a, **k: called.setdefault("s", True)
    )
    fig, ax = expe.plot_resistance(t_arr, show=True)
    assert called.get("s", False)
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
