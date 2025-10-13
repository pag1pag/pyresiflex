import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.experiment.purely_resistive_experiment import (
    PurelyResistiveExperiment,
)
from pyresiflex.generator.generator_real_impedance import TrapezoidalGenerator
from pyresiflex.load.time_varying_resistance import PlasmaResistanceLinearFall
from pyresiflex.misc.utils import get_path_to_data
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

plt.style.use(get_path_to_data("article_two_columns_figure.mplstyle"))


@pytest.fixture
def setup_experiment() -> dict[str, object]:
    """Set up a purely resistive experiment with a plasma load.

    Returns
    -------
    dict[str, object]
        A dictionary containing the setup parameters and results.
    """
    # Transmission solution parameters
    L = 0.3
    c = 1.66e8
    Z_c = 75
    cable = PerfectCable(L=L, c=c, Z_c=Z_c)

    # Generator parameters
    R_g = 1
    U_off = 0.0
    U_on = 10e3
    t_rise = 3e-9
    t_on = 10e-9
    t_fall = 3e-9
    generator = TrapezoidalGenerator(
        R_g=R_g,
        U_off=U_off,
        U_on=U_on,
        t_rise=t_rise,
        t_on=t_on,
        t_fall=t_fall,
    )

    # Load parameters
    Z_start = 1e3
    Z_end = 10
    t_start_fall = L / c + 5e-9
    t_end_fall = L / c + 10e-9
    plasma_load = PlasmaResistanceLinearFall(
        Z_start=Z_start,
        Z_end=Z_end,
        t_start_fall=t_start_fall,
        t_end_fall=t_end_fall,
    )

    solution = PurelyResistiveSolution(
        generator=generator,
        load=plasma_load,
        cable=cable,
    )

    times = np.arange(0, 30, 0.1) * 1e-9
    x_meas_voltage = 2 / 3 * L
    voltages = np.array([solution.V(x_meas_voltage, t) for t in times])
    x_meas_current = 2 / 3 * L
    currents = np.array([solution.I(x_meas_current, t) for t in times])

    return {
        "L": L,
        "c": c,
        "Z_c": Z_c,
        "generator": generator,
        "plasma_load": plasma_load,
        "solution": solution,
        "times": times,
        "x_meas_voltage": x_meas_voltage,
        "voltages": voltages,
        "x_meas_current": x_meas_current,
        "currents": currents,
    }


def test_reconstruct_resistance_from_signals(
    setup_experiment,
    plot: bool = False,
):
    # Load the experiment setup.
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=setup_experiment["times"],
        experimental_voltage_value=setup_experiment["voltages"],
        x_meas_voltage=setup_experiment["x_meas_voltage"],
        experimental_current_time=setup_experiment["times"],
        experimental_current_value=setup_experiment["currents"],
        x_meas_current=setup_experiment["x_meas_current"],
        L=setup_experiment["L"],
        Z_c=setup_experiment["Z_c"],
        c=setup_experiment["c"],
        correct_time_zero=False,
    )

    # Get the true resistance for comparison.
    true_resistance = np.array(
        [
            setup_experiment["plasma_load"].load_impedance(t)
            for t in setup_experiment["times"]
        ]
    )

    # Compute R_p(vmes, imes)
    expe.compute_plasma_resistance_from_vmes_and_imes(
        setup_experiment["times"],
        threshold=1e-6,
    )
    resistance_v_i = expe.Rp_corrected_with_nan
    assert resistance_v_i is not None

    # Compute R_p(vmes, vg)
    resistance_v_vg = expe.compute_plasma_resistance_from_vmes_and_vg(
        setup_experiment["times"],
        generator=setup_experiment["generator"],
        max_n=4,
    )
    assert resistance_v_vg is not None

    # Compute R_p(imes, vg)
    resistance_i_vg = expe.compute_plasma_resistance_from_imes_and_vg(
        setup_experiment["times"],
        generator=setup_experiment["generator"],
        max_n=4,
    )
    assert resistance_i_vg is not None

    # .. Check the L2 norm of the difference.

    def check_norms(resistance: np.ndarray, true_resistance: np.ndarray):
        # Some values can be NaN, so we ignore them in the comparison.
        mask_nan = ~np.isnan(resistance) & ~np.isnan(true_resistance)
        # Furthermore, before the reflected wave reaches the measurement
        # point, the resistance cannot be reconstructed, so we ignore
        # these values as well.
        # They correspond to times t < L/c + (L - x_meas_voltage)/c
        L = setup_experiment["L"]
        c = setup_experiment["c"]
        x_meas_voltage = setup_experiment["x_meas_voltage"]
        mask_before_reflection = (
            setup_experiment["times"] >= L / c + (L - x_meas_voltage) / c
        )
        mask = mask_nan & mask_before_reflection

        norm = np.linalg.norm(
            resistance[mask] - true_resistance[mask],
            ord=2,
        ) / np.linalg.norm(true_resistance[mask], ord=2)
        return norm

    assert np.isclose(
        check_norms(resistance_v_i, true_resistance),
        0,
        atol=1e-2,
    )
    assert np.isclose(
        check_norms(resistance_v_vg, true_resistance),
        0,
        atol=1e-2,
    )
    assert np.isclose(
        check_norms(resistance_i_vg, true_resistance),
        0,
        atol=1e-2,
    )

    # Optionally plot the results.
    if plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(
            setup_experiment["times"] * 1e9,
            true_resistance,
            color="lightgray",
            label="True Resistance",
        )
        ax.plot(
            setup_experiment["times"] * 1e9,
            resistance_v_i,
            label="R_p(v_meas, i_meas)",
        )
        ax.plot(
            setup_experiment["times"] * 1e9,
            resistance_v_vg,
            label="R_p(v_meas, v_g)",
        )
        ax.plot(
            setup_experiment["times"] * 1e9,
            resistance_i_vg,
            label="R_p(i_meas, v_g)",
        )
        ax.axvline(
            setup_experiment["plasma_load"].t_start_fall * 1e9,
            color="k",
            linestyle="--",
            label="Start of fall",
        )
        ax.axvline(
            setup_experiment["plasma_load"].t_end_fall * 1e9,
            color="r",
            linestyle="--",
            label="End of fall",
        )
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Plasma Resistance (Ohm)")
        ax.set_yscale("log")
        ax.set_title("Reconstructed Plasma Resistance")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
