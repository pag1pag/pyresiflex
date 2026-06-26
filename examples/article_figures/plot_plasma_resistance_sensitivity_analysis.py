r"""
Extracting plasma resistance and analyzing sensitivity to perturbations.
========================================================================

In this example, the plasma is assumed to be a time-varying resistance.

Experimental data of [Minesi2022]_ are used to extract the plasma resistance.
For the remote configuration case, voltage and current signals are measured at
the middle of a cable of length :math:`L \approx 6 \, \mathrm{m}`.

Starting with a baseline scenario, the plasma resistance is computed and then
perturbed for sensitivity analysis.
Then, using this reconstructed plasma resistance, voltage and current signals
are simulated, and compared against the measured signals.
"""  # noqa: D205

# This sets the third figure as the thumbnail for the example gallery.
# sphinx_gallery_thumbnail_number = 3
# This displays each image separately in the example gallery.
# sphinx_gallery_multi_image = "single"

# %%
# First, we import the required libraries.
# -----------------------------------------
#
# We start by importing the modules we need:
#
# - matplotlib for drawing graphs,
# - numpy for array functions,
# - pyresiflex for the generator, load and transmission line.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.experiment.purely_resistive_experiment import (
    PurelyResistiveExperiment,
)
from pyresiflex.generator.generator_real_impedance import (
    FromMeasurementGenerator,
)
from pyresiflex.misc.load_data import load_minesi_data
from pyresiflex.misc.plot import plot_voltage_current, set_mpl_style
from pyresiflex.misc.utils import get_path_to_data
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

set_mpl_style(nb_columns=2)

# %%
# Load [Minesi2022]_ experimental data of remote configuration.
# -------------------------------------------------------------

# Load the raw data from Figure 16 of [Minesi2022]_.
file = get_path_to_data(
    "Minesi2022",
    "fig16_remoteConfiguration.csv",
)
data = np.loadtxt(file, skiprows=3, delimiter=";")
times_raw = data[:, 0] * 1e-9  # [s]
voltages_raw = data[:, 1] * 1e3  # [V]
currents_raw = data[:, 3]  # [A]

# Plot the raw data.
fig, ax_v, ax_i = plot_voltage_current(
    voltage_time=times_raw,
    voltage_value=voltages_raw,
    current_time=times_raw,
    current_value=currents_raw,
)
ax_v.set_xlabel(r"$\mathregular{t - \frac{x_{meas}}{c} \, [ns]}$")
ax_v.set_ylim(-4, 4)
ax_i.set_ylim(-60, 60)

plt.show()

# %%
# Preprocess the data.
# --------------------

# Define the zero at the first time the voltage reaches `threshold_voltage`.
threshold_voltage = 25  # [V]
idx_first = np.where(np.abs(voltages_raw) > threshold_voltage)[0][0]
times_raw = times_raw - times_raw[idx_first]

# Define a time window to analyze.
lower_time_window = -20e-9  # [s]
upper_time_window = 200e-9  # [s]

# Limit the time window to [lower_time_window, upper_time_window]
idx_min_wanted_time = np.where(times_raw > lower_time_window)[0][0]
idx_max_wanted_time = np.where(times_raw > upper_time_window)[0][0]

# Limit the time, voltages and currents to the wanted period.
times_expe = times_raw[idx_min_wanted_time:idx_max_wanted_time]
voltages_expe = voltages_raw[idx_min_wanted_time:idx_max_wanted_time]
currents_expe = currents_raw[idx_min_wanted_time:idx_max_wanted_time]

# Plot the preprocessed data.
fig, ax_v, ax_i = plot_voltage_current(
    voltage_time=times_expe,
    voltage_value=voltages_expe,
    current_time=times_expe,
    current_value=currents_expe,
)
ax_v.set_xlabel(r"$\mathregular{t - \frac{x_{meas}}{c} \, [ns]}$")
ax_v.set_xlim(0, 200)
ax_v.set_ylim(-4, 4)
ax_i.set_ylim(-60, 60)
ax_i.set_yticks([-60, -45, -30, -15, 0, 15, 30, 45, 60])

plt.show()

# %%
# Define the scenarios for the sensitivity analysis.
# --------------------------------------------------

# Transmission line parameters estimated from experimental data.
# See `plot_determine_Minesi2022_parameters.py` example for more details.
data = load_minesi_data()

# Default scenario:
# Length of the transmission line
default_L = data.L  # [m]
# Measurement points = probe positions
default_x = data.x_meas  # [m]
# Velocity of propagation of the wave in the cable.
default_c = data.c  # [m/s]
# Cable characteristic impedance.
default_Z_c = data.Z_c  # [Ohm]


# Dict of scenarios:
# - key: scenario name
# - value: dict of parameters
#   - key: parameter to vary
#   - value: scaling factor (1.0 is unchanged)
scenarios = {
    "Baseline": {
        "L": 1.0,
        "x": 1.0,
        "c": 1.0,
        "Z_c": 1.0,
        "color": "black",
        "channel_formation_time": 42e-9,
    },
    "L * 0.9": {
        "L": 0.9,
        "x": 1.0,
        "c": 1.0,
        "Z_c": 1.0,
        "color": "tab:red",
        "channel_formation_time": 46e-9,
    },
    "L * 1.1": {
        "L": 1.1,
        "x": 1.0,
        "c": 1.0,
        "Z_c": 1.0,
        "color": "tab:blue",
        "channel_formation_time": 39e-9,
    },
    "x * 0.9": {
        "L": 1.0,
        "x": 0.9,
        "c": 1.0,
        "Z_c": 1.0,
        "color": "tab:red",
        "channel_formation_time": 39e-9,
    },
    "x * 1.1": {
        "L": 1.0,
        "x": 1.1,
        "c": 1.0,
        "Z_c": 1.0,
        "color": "tab:blue",
        "channel_formation_time": 45e-9,
    },
    "c * 0.9": {
        "L": 1.0,
        "x": 1.0,
        "c": 0.9,
        "Z_c": 1.0,
        "color": "tab:red",
        "channel_formation_time": 39e-9,
    },
    "c * 1.1": {
        "L": 1.0,
        "x": 1.0,
        "c": 1.1,
        "Z_c": 1.0,
        "color": "tab:blue",
        "channel_formation_time": 42e-9,
    },
    "Z_c * 0.9": {
        "L": 1.0,
        "x": 1.0,
        "c": 1.0,
        "Z_c": 0.9,
        "color": "tab:red",
        "channel_formation_time": 42e-9,
    },
    "Z_c * 1.1": {
        "L": 1.0,
        "x": 1.0,
        "c": 1.0,
        "Z_c": 1.1,
        "color": "tab:blue",
        "channel_formation_time": 42e-9,
    },
}

# %%
# Plot plasma resistance, voltage and current signals from Minesi2022.
# --------------------------------------------------------------------

# Figure and ax for the plasma resistance.
fig_r, [[ax_r_L, ax_r_x], [ax_r_c, ax_r_Z]] = plt.subplots(
    nrows=2, ncols=2, figsize=(16 * 2, 9 * 2)
)
axes_r: list[Axes] = [ax_r_L, ax_r_x, ax_r_c, ax_r_Z]

# Figure and ax for the voltage.
fig_v, [[ax_v_L, ax_v_x], [ax_v_c, ax_v_Z]] = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(16 * 2, 9 * 2),
)
axes_v: list[Axes] = [ax_v_L, ax_v_x, ax_v_c, ax_v_Z]
# Plot the measured voltage.
for ax_v in axes_v:
    plot_line_v_measured = ax_v.plot(
        (times_expe + default_x / default_c) * 1e9,
        voltages_expe * 1e-3,
        color="gray",
        label="Voltage, experimental",
    )

# Figure and ax for the current.
fig_i, [[ax_i_L, ax_i_x], [ax_i_c, ax_i_Z]] = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(16 * 2, 9 * 2),
)
axes_i: list[Axes] = [ax_i_L, ax_i_x, ax_i_c, ax_i_Z]
# Plot the measured current.
for ax_i in axes_i:
    plot_line_i_measured = ax_i.plot(
        (times_expe + default_x / default_c) * 1e9,
        currents_expe,
        color="gray",
        label="Current, experimental",
    )


for scenario, params in scenarios.items():
    # Apply scaling factors for each scenario.
    assert isinstance(params["L"], float)
    L = default_L * params["L"]
    assert isinstance(params["x"], float)
    x = default_x * params["x"]
    assert isinstance(params["c"], float)
    c = default_c * params["c"]
    assert isinstance(params["Z_c"], float)
    Z_c = default_Z_c * params["Z_c"]
    assert isinstance(params["channel_formation_time"], float)
    channel_formation_time = params["channel_formation_time"]

    print(f"Scenario: {scenario}")
    print("-----------------------------")
    print(f"L = {L:.2f} m, x = {x:.2f} m")
    print(f"Wave velocity: {c:.2e} m/s")
    print(f"Characteristic impedance: {Z_c:.2f} Ohm")
    print("\n\n")

    # .. Plasma load.
    expe = PurelyResistiveExperiment(
        experimental_voltage_time=times_expe,
        experimental_voltage_value=voltages_expe,
        x_meas_voltage=x,
        experimental_current_time=times_expe,
        experimental_current_value=currents_expe,
        x_meas_current=x,
        L=L,
        Z_c=Z_c,
        c=c,
        correct_time_zero=True,
    )

    expe.compute_plasma_resistance_from_vmeas_and_imeas(
        times_expe,
        threshold=400,
        channel_formation_time=channel_formation_time,
    )

    # Without correction:
    expected_R_p_no_corr = expe.R_p  # [Ohm]
    expected_times_no_corr = expe.times  # [s]
    # With correction:
    expected_R_p_with_corr = expe.Rp_corrected  # [Ohm]
    expected_times_with_corr = expe.times_corrected  # [s]
    # With correction (and nan):
    expected_R_p_with_corr_nan = expe.Rp_corrected_with_nan  # [Ohm]
    expected_times_with_corr_nan = expe.times_corrected_with_nan  # [s]

    # We use the corrected resistance and times for the plasma load.
    plasma_load = expe.load_corrected

    if scenario.startswith("Baseline"):
        for ax_r in axes_r:
            # Plot in light color the resistance without time correction.
            ax_r.plot(
                expected_times_no_corr * 1e9,
                expected_R_p_no_corr,
                color="lightgray",
            )
            # Then, plot the resistance with time correction.
            ax_r.plot(
                expected_times_with_corr_nan * 1e9,
                expected_R_p_with_corr_nan,
                color=params["color"],
                ls="-",
            )
            ax_r.plot(
                expected_times_with_corr * 1e9,
                expected_R_p_with_corr,
                color=params["color"],
                ls="--",
                lw=2,
            )
    else:
        if scenario.startswith("L"):
            ax_r = ax_r_L
        elif scenario.startswith("c"):
            ax_r = ax_r_c
        elif scenario.startswith("x"):
            ax_r = ax_r_x
        elif scenario.startswith("Z"):
            ax_r = ax_r_Z
        else:
            raise ValueError("Invalid scenario.")
        # Plot in light color the resistance without time correction.
        ax_r.plot(
            expected_times_no_corr * 1e9,
            expected_R_p_no_corr,
            color=params["color"],
            alpha=0.2,
        )
        # Then, plot the resistance with time correction.
        ax_r.plot(
            expected_times_with_corr_nan * 1e9,
            expected_R_p_with_corr_nan,
            color=params["color"],
            ls="-",
        )
        ax_r.plot(
            expected_times_with_corr * 1e9,
            expected_R_p_with_corr,
            color=params["color"],
            ls="--",
            lw=2,
        )

    # Reconstruct voltage and current with resistance.
    # ------------------------------------------------

    # .. Generator parameters.
    # Impedance of the generator.
    R_g = data.R_g  # [Ohm]
    # Attenuation coefficient.
    alpha_g = data.alpha_g  # [-]
    # Pulse duration.
    pulse_duration = 35e-9  # [s]

    def V_meas_generator(t, times, voltages):
        if t < 0:
            return 0.0
        elif t > pulse_duration:
            return 0.0
        else:
            return np.interp(t, times, voltages) / alpha_g

    generator = FromMeasurementGenerator(
        R_g=R_g,
        V_meas=lambda t: V_meas_generator(t, times_expe, voltages_expe),
    )

    # .. Transmission line.
    cable = PerfectCable(
        L=L,
        Z_c=Z_c,
        c=c,
    )

    # .. Solution.
    solution = PurelyResistiveSolution(
        generator=generator,
        load=plasma_load,
        cable=cable,
    )
    # Time vector
    N = 1000  # Number of time points.
    times = np.linspace(lower_time_window, upper_time_window, N)  # [s]
    # Compute the voltage and current at a given position.
    solution.solve(x, times)

    # Get the voltage and current signals.
    voltages = solution.voltage  # [V]
    currents = solution.current  # [A]

    # Plot voltage.
    if scenario.startswith("Baseline"):
        for ax_v in axes_v:
            ax_v.plot(
                times * 1e9,
                voltages * 1e-3,
                color=params["color"],
                ls="dashed",
                label=f"Voltage, {scenario}",
            )
        for ax_i in axes_i:
            ax_i.plot(
                times * 1e9,
                currents,
                color=params["color"],
                ls="dashed",
                label=f"Current, {scenario}",
            )
    elif scenario.startswith("L"):
        ax_v = ax_v_L
        ax_i = ax_i_L
    elif scenario.startswith("c"):
        ax_v = ax_v_c
        ax_i = ax_i_c
    elif scenario.startswith("x"):
        ax_v = ax_v_x
        ax_i = ax_i_x
    elif scenario.startswith("Z"):
        ax_v = ax_v_Z
        ax_i = ax_i_Z
    else:
        raise ValueError("Invalid scenario.")

    plot_line_v = ax_v.plot(
        times * 1e9,
        voltages * 1e-3,
        color=params["color"],
        ls="dashed",
        label=f"Voltage, {scenario}",
    )
    plot_line_i = ax_i.plot(
        times * 1e9,
        currents,
        color=params["color"],
        ls="dashed",
        label=f"Current, {scenario}",
    )


# Plot options...

# .. for resistance
for ax_r in axes_r:
    ax_r.set_xlabel(r"$\mathregular{t \, [ns]}$")
    ax_r.set_ylabel(r"$\mathregular{R_p \, [\Omega]}$")
    ax_r.set_xlim(30, 80)
    ax_r.set_ylim(-50, 1000)

    # Add legend
    (line_r_no_corr_legend,) = ax_r.plot(
        [],
        [],
        color="lightgray",
        linestyle="-",
    )
    (line_r_with_corr_legend,) = ax_r.plot(
        [],
        [],
        color="black",
        ls="-",
    )
    (line_r_with_corr_legend_interpolation,) = ax_r.plot(
        [],
        [],
        color="black",
        ls="--",
        lw=2,
    )
    ax_r.legend(
        [
            line_r_no_corr_legend,
            line_r_with_corr_legend,
            line_r_with_corr_legend_interpolation,
        ],
        [
            "Resistance (no corr.)",
            "Resistance (with corr.)",
            "Resistance (interpolated)",
        ],
        labelcolor="black",
        loc="upper right",
    )


# .. for voltage
for ax_v in axes_v:
    ax_v.set_xlabel(r"$\mathregular{t \, [ns]}$")
    ax_v.set_ylabel(r"$\mathregular{V(x=\frac{L}{2}, t) \, [kV]}$")
    ax_v.set_xlim(0, 150)
    ax_v.set_ylim(-4, 4)

    # # Add legend
    # (line_V_meas_legend,) = ax_v.plot(
    #     [],
    #     [],
    #     color="lightgray",
    #     linestyle="-",
    # )
    # (line_v_model_legend,) = ax_v.plot(
    #     [],
    #     [],
    #     color="black",
    #     ls="--",
    # )
    # ax_v.legend(
    #     [line_V_meas_legend, line_v_model_legend],
    #     ["Measurement", "Model"],
    #     labelcolor="black",
    #     loc="upper right",
    # )

# .. for current
for ax_i in axes_i:
    ax_i.set_xlabel(r"$\mathregular{t \, [ns]}$")
    ax_i.set_ylabel(r"$\mathregular{I(x=\frac{L}{2}, t) \, [kA]}$")
    ax_i.set_xlim(0, upper_time_window * 1e9)
    ax_i.set_ylim(-60, 60)

plt.show()

# %%
# Save the figure.
# ----------------

# Export the image to a .svg file, in the figures folder.
fig_r.savefig(
    get_path_to_data(
        "article_figures",
        "Minesi2022_sensitivity_reconstruction_plasma_resistance.svg",
        force_return=True,
    ),
)
fig_v.savefig(
    get_path_to_data(
        "article_figures",
        "Minesi2022_sensitivity_reconstruction_voltage.svg",
        force_return=True,
    ),
)

# %%
