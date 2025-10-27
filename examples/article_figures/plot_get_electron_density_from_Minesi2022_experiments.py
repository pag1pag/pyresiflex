r"""
Compare electron density results between electrical and optical methods.
========================================================================

This example shows how to compute electron density in a plasma from
electrical measurements, and compare it with experimental data obtained via
Optical Emission Spectroscopy (OES).


In this example, the plasma is assumed to be a time-varying resistance.

Experimental data of [Minesi2022]_ are used to extract the plasma resistance,
as well as the electric field in the plasma.

This electric field is then used to get the electron mobility vs time. This
mobility was computed with Bolsig+ beforehand, versus reduced electric field.

The electron density is then computed from the plasma resistance and the
mobility, and compared to experimental data from OES measurements provided
by [Minesi2022]_.
"""  # noqa: D205

# This sets the third figure as the thumbnail for the example gallery.
# sphinx_gallery_thumbnail_number = 13
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

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.experiment.purely_resistive_experiment import (
    PurelyResistiveExperiment,
)
from pyresiflex.generator.generator_real_impedance import (
    FromMeasurementGenerator,
)
from pyresiflex.misc.plot import plot_voltage_current, set_mpl_style
from pyresiflex.misc.units import e, k_b
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
raw_data = np.loadtxt(file, skiprows=3, delimiter=";")
times_raw = raw_data[:, 0] * 1e-9  # [s]
voltages_raw = raw_data[:, 1] * 1e3  # [V]
currents_raw = raw_data[:, 3]  # [A]

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

# Compute the energy from the voltage and current.
energies_expe = np.zeros_like(times_expe)  # [J]
for i in range(len(times_expe)):
    energies_expe[i] = np.trapezoid(
        voltages_expe[:i] * currents_expe[:i], times_expe[:i]
    )


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
# Transmission line parameters.
# -----------------------------

# Transmission line parameters estimated from experimental data.
# See `plot_determining_cable_properties.py` example for more details.

# Length of the transmission line
L = 6.2  # [m]
# Measurement points = probe positions
x = L / 2  # [m]
# Here, we assume that the probes are located at the same position.
x_meas_voltage = x_meas_current = x  # [m]
# Velocity of propagation of the wave in the cable.
c = 1.78e8  # [m/s]
# Cable characteristic impedance.
Z_c = 69  # [Ohm]

cable = PerfectCable(
    L=L,
    Z_c=Z_c,
    c=c,
)


# %%
# Generator parameters.
# ---------------------

# Impedance of the generator.
R_g = 10  # [Ohm]  (Chosen between 1 and 10 Ohm in the article)
# Attenuation coefficient.
alpha_g = Z_c / (Z_c + R_g)  # [-]
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
    R_g=R_g, V_meas=lambda t: V_meas_generator(t, times_expe, voltages_expe)
)

# Plot the voltage signal.
set_mpl_style(nb_columns=1)
fig, ax = plt.subplots()
ax.plot(
    times_expe * 1e9,
    voltages_expe * 1e-3,
    color="black",
    label="Measured voltage",
)
ax.plot(
    times_expe * 1e9,
    [generator.generator_voltage(t) * 1e-3 for t in times_expe],
    "--",
    label="Model generator",
    color="red",
)
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{V \, [kV]}$")
ax.set_xlim(0, 50)
ax.set_ylim(-0.1, 4.5)
ax.legend()
plt.show()


# %%
# Plasma parameters.
# ------------------

expe = PurelyResistiveExperiment(
    experimental_voltage_time=times_expe,
    experimental_voltage_value=voltages_expe,
    x_meas_voltage=x_meas_voltage,
    experimental_current_time=times_expe,
    experimental_current_value=currents_expe,
    x_meas_current=x_meas_current,
    L=L,
    Z_c=Z_c,
    c=c,
    correct_time_zero=True,
)

threshold_voltage_for_resistance = 0.2 * np.max(voltages_expe)  # [V]
expe.compute_plasma_resistance_from_vmes_and_imes(
    times_expe,
    threshold=threshold_voltage_for_resistance,
    channel_formation_time=42e-9,
)

plasma_load = expe.load_corrected

# Plot the plasma resistance.
set_mpl_style(nb_columns=1)
fig, ax = expe.plot_resistance(times=times_expe)
ax.set_xlim(0, 200)
ax.set_ylim(-100, 1000)
plt.show()


# %%
# Solution object.
# ----------------

solution = PurelyResistiveSolution(
    generator=generator,
    load=plasma_load,
    cable=cable,
)


# %%
# Compute voltage, current and energy at plasma position.
# -------------------------------------------------------

# Time vector for the simulation.
nb_steps = 1000
times = np.linspace(lower_time_window, upper_time_window, nb_steps)  # [s]
# Position of probes for plasma position.
x = L  # [m]

# Compute the voltage and current at plasma position.
solution.solve(x, times)

voltages = solution.voltage  # [V]
currents = solution.current  # [A]
energies = solution.energy  # [J]
xs = solution.x  # [m]
times = solution.t  # [s]


# %%
# Plot voltage, current, and energy at plasma.
# --------------------------------------------

# Do we want to plot the current and energy?
plot_current = True
plot_energy = True
# Do we want to shift the time axis to have t - x/c?
shift_time_axis = False

if shift_time_axis:
    times_shifted = times - x / c
    times_expe_shifted = times_expe
    x_label = r"$\mathregular{t - \frac{x_{meas}}{c} \, [ns]}$"
else:
    times_shifted = times
    times_expe_shifted = times_expe + x / c
    x_label = r"$\mathregular{t \, [ns]}$"

fig, ax_v = plt.subplots()

# .. Set title.
suptitle = "Voltage"
if plot_current and plot_energy:
    suptitle += ", current and energy"
elif plot_current:
    suptitle += " and current"
elif plot_energy:
    suptitle += " and energy"
suptitle += f" at x = {x:.2f} m"

# Plot voltage.
plot_line_v = ax_v.plot(
    times_shifted * 1e9,
    voltages * 1e-3,
    color="k",
    ls="-",
    label="Voltage (computed)",
)
# .. Plot options for voltage.
ax_v.set_xlabel(x_label)
ax_v.set_ylabel(r"$\mathregular{V \, [kV]}$")
ax_v.set_ylim(-8, 8)
ax_v.spines["left"].set_color("k")
ax_v.set_xlim(0, times_shifted[-1] * 1e9)

# Plot current.
if plot_current:
    ax_i = ax_v.twinx()
    ax_i.plot(
        times_shifted * 1e9,
        currents,
        color="r",
        ls="-",
        label="Current (computed)",
    )
    # .. Plot options for current.
    ax_i.set_ylabel(r"$\mathregular{I \, [A]}$", color="r")
    # ax_i.set_ylim(-max_abs_current, max_abs_current)
    ax_i.set_ylim(-60, 60)
    ax_i.grid(visible=False)
    # Change color of the right y-axis to red.
    ax_i.spines["right"].set_color("r")
    # Also change the color of the ticks.
    ax_i.tick_params(axis="y", colors="r")
    # Move x-position of the y-label.
    ax_i.yaxis.set_label_coords(1.05, 0.5)
    # Set y-ticks for current.
    ax_i.set_yticks([-60, -45, -30, -15, 0, 15, 30, 45, 60])

# Plot energy.
if plot_energy:
    ax_e = ax_v.twinx()
    ax_e.plot(
        times_shifted * 1e9,
        energies * 1e3,
        color="b",
        ls="-",
        label="Energy",
    )
    # .. Plot options for energy.
    ax_e.set_ylabel(r"$\mathregular{E \, [mJ]}$", color="b")
    # Move the y-axis of ax_e to the right, by 100 points
    ax_e.spines["right"].set_position(("outward", 100))
    ax_e.grid(visible=False)
    ax_e.set_ylim(0, 2.4)
    ax_e.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4])
    # Change color of the right y-axis to blue.
    ax_e.spines["right"].set_color("b")
    # Also change the color of the ticks.
    ax_e.tick_params(axis="y", colors="b")

ax_v.legend(
    handles=plot_line_v,
    labels=["Model"],
    loc="lower right",
)

plt.show()


# %%
# Plot reduced electric field in the plasma.
# ------------------------------------------

times_plasma = np.copy(times_shifted)
plasma_voltage = np.copy(voltages)

# Plot the plasma voltage.
fig, ax = plt.subplots()
ax.plot(times_plasma * 1e9, plasma_voltage * 1e-3)
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{V_{plasma} \, [kV]}$")
ax.set_title("Extracted plasma voltage")
ax.set_ylim(-4, 7)
plt.show()

# Compute the reduced electric field in the plasma.
P = 1.013e5  # [Pa], atmospheric pressure
gap = 5e-3  # [m], discharge length (cf. [Minesi2022]_)
plasma_electric_field = np.abs(plasma_voltage) / gap  # [V/m], absolute
n_gas_2000K = P / (k_b * 2000)  # [m^-3], neutral density at 1 atm and 2000 K.
n_gas_3000K = P / (k_b * 3000)  # [m^-3], neutral density at 1 atm and 3000 K.
E_N_2000K = plasma_electric_field / n_gas_2000K  # [V m^2]
E_N_3000K = plasma_electric_field / n_gas_3000K  # [V m^2]
E_N_2000K_Td = E_N_2000K * 1e21  # [Td]
E_N_3000K_Td = E_N_3000K * 1e21  # [Td]

# Plot the reduced electric field.
fig, ax = plt.subplots()
ax.plot(times_plasma * 1e9, E_N_2000K_Td, label="T=2000 K")
ax.plot(times_plasma * 1e9, E_N_3000K_Td, label="T=3000 K")
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{\frac{|E|}{N} \, [Td]}$")
ax.set_title("(Absolute) reduced electric field in the plasma")
ax.legend()
plt.show()

# %%
# Get mobility from Bolsig+.
# --------------------------

# Load Bolsig+ data.
all_data: dict[str, dict[str, np.ndarray]] = {
    "2000K_Minesi2022.csv": {},
    "3000K_Minesi2022.csv": {},
    # "3000K_ee_ei__ngas_2.5e24__ionisation_1e-4": {},
    # "3000K_ee_ei__ngas_2.5e24__ionisation_1e-3": {},
    # "3000K_ee_ei__ngas_2.5e24__ionisation_1e-2": {},
    # "3000K_ee_ei__ngas_2.5e24__ionisation_1e-1": {},
}
plot_options = {
    "2000K_Minesi2022.csv": {
        "color": "magenta",
        "linestyle": "-",
        "label": "2000 K",
    },
    "3000K_Minesi2022.csv": {
        "color": "orange",
        "linestyle": "-",
        "label": "3000 K",
    },
    # "3000K_ee_ei__ngas_2.5e24__ionisation_1e-3": {
    #     "color": "blue",
    #     "linestyle": "--",
    #     "label": r"3000 K (ee/ei, $\alpha$=1e-3)",
    # },
    # "3000K_ee_ei__ngas_2.5e24__ionisation_1e-2": {
    #     "color": "green",
    #     "linestyle": "--",
    #     "label": r"3000 K (ee/ei, $\alpha$=1e-2)",
    # },
}

for file_name in all_data:
    output_folder = get_path_to_data(
        "Minesi2022",
        "cross_sections",
        "results",
        file_name
    )

    bolsig_data = np.loadtxt(output_folder, delimiter=",", skiprows=1)
    E_N_Td = bolsig_data[:, 0]  # [Td]
    mu_N = bolsig_data[:, 1]  # [1/(m*V*s)]

    T = int(file_name.split("K")[0])  # [K]
    n_gas = P / (k_b * T)  # [m^-3]

    all_data[file_name]["E_N_Td_bolsig"] = E_N_Td  # [Td]
    all_data[file_name]["mu_N_bolsig"] = mu_N  # [1/(m*V*s)]
    all_data[file_name]["mu_bolsig"] = mu_N / n_gas  # [m^2/(V*s)]

# Plot mobility * N from Bolsig+.
fig_E, ax_E = plt.subplots()
for file_name, data in all_data.items():
    label = file_name.replace("_", " ")
    ax_E.plot(
        data["E_N_Td_bolsig"],
        data["mu_N_bolsig"],
        label=plot_options[file_name]["label"],
        color=plot_options[file_name]["color"],
        linestyle=plot_options[file_name]["linestyle"],
    )
ax_E.set_title("Mobility * N from Bolsig+")
ax_E.set_xlabel(r"$\mathregular{E/N \, [Td]}$")
ax_E.set_ylabel(r"$\mathregular{\mu \cdot N \, [1/(m \cdot V \cdot s)]}$")
ax_E.set_xscale("log")
ax_E.legend(fontsize=20)

# Plot mobility from Bolsig+.
fig_E, ax_E = plt.subplots()
for file_name, data in all_data.items():
    label = file_name.replace("_", " ")
    ax_E.plot(
        data["E_N_Td_bolsig"],
        data["mu_bolsig"],
        label=plot_options[file_name]["label"],
        color=plot_options[file_name]["color"],
        linestyle=plot_options[file_name]["linestyle"],
    )
ax_E.set_title("Mobility from Bolsig+")
ax_E.set_xlabel(r"$\mathregular{E/N \, [Td]}$")
ax_E.set_ylabel(r"$\mathregular{\mu \, [m^2/(V \cdot s)]}$")
ax_E.set_xscale("log")
ax_E.legend(fontsize=20)

plt.show()

# %%
# Plot the mobility vs time.
# --------------------------

# Apply the reduced electric field in the plasma to get the mobility vs time.
for file_name in all_data:
    # Get the temperature from the folder name.
    T = int(file_name.split("K")[0])  # [K]
    # Compute the gas density.
    n_gas = P / (k_b * T)  # [m^-3]
    # Compute the reduced electric field in the plasma.
    E_N_Td = (plasma_electric_field / n_gas) * 1e21  # [Td]

    # Interpolate the mobility vs reduced electric field.
    E_N_bolsig = all_data[file_name]["E_N_Td_bolsig"]
    mu_bolsig = all_data[file_name]["mu_bolsig"]
    mu_interpolated = np.interp(
        E_N_Td,
        E_N_bolsig,
        mu_bolsig,
        left=np.nan,
        right=np.nan,
    )

    # Add the interpolated mobility to the data dictionary.
    all_data[file_name]["E_N_Td_plasma"] = E_N_Td
    all_data[file_name]["mu_interpolated"] = mu_interpolated

# Plot the mobility vs time.
fig, ax = plt.subplots()
for file_name, data in all_data.items():
    label = file_name.replace("_", " ")
    mu_interpolated = data["mu_interpolated"]
    ax.plot(
        times_plasma * 1e9,
        mu_interpolated,
        label=plot_options[file_name]["label"],
        color=plot_options[file_name]["color"],
        linestyle=plot_options[file_name]["linestyle"],
    )
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{\mu \, [m^2/(V*s)]}$")
ax.set_title("Mobility in the plasma")
ax.legend(fontsize=20)
ax.set_xlim(0, 200)
plt.show()

# %%
# Plot the resistance vs time.
# ----------------------------

# Get the expected plasma resistance vs time.
times_expe = expe.times_corrected
times_expe_with_nan = expe.times_corrected_with_nan
expected_R_p = expe.Rp_corrected
expected_R_p_with_nan = expe.Rp_corrected_with_nan

# Plot the expected plasma resistance vs time.
fig, ax = plt.subplots()
ax.plot(
    times_expe_with_nan * 1e9,
    expected_R_p_with_nan,
    label="Extracted plasma resistance",
)
ax.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
ax.set_ylabel(r"$\mathregular{R_{plasma} \, [\Omega]}$")
ax.set_title("Extracted plasma resistance vs time")
ax.set_ylim(-100, 1000)
ax.set_xlim(0, 200)
plt.show()


# %%
# Compute electron density vs time.
# ---------------------------------

gap = 5e-3  # [m], discharge length
r_discharge = 0.6e-3  # [m], discharge radius
S_discharge = np.pi * r_discharge**2  # [m^2], discharge cross-section area

R_p_interpolated_with_nan = np.interp(
    times_plasma,
    times_expe_with_nan,
    expected_R_p_with_nan,
    left=np.nan,
    right=np.nan,
)

for file_name in all_data:
    # Compute electron density vs time.
    mu_interpolated = all_data[file_name]["mu_interpolated"]
    n_e_with_nan = (
        gap  # [m]
        / S_discharge  # [m^2]
        / e  # [C] = [A*s]
        / mu_interpolated  # [m^2/(V*s)]
        / R_p_interpolated_with_nan  # [Ohm] = [V/A]
    )  # [m^-3]
    all_data[file_name]["n_e_with_nan"] = n_e_with_nan

    # Compute density, with discharge radius divided by 2.
    r_discharge_half = r_discharge / 2  # [m]
    S_discharge_half = np.pi * r_discharge_half**2  # [m^2]
    n_e_half_with_nan = (
        gap  # [m]
        / S_discharge_half  # [m^2]
        / e  # [C] = [A*s]
        / mu_interpolated  # [m^2/(V*s)]
        / R_p_interpolated_with_nan  # [Ohm] = [V/A]
    )  # [m^-3]
    all_data[file_name]["n_e_half_with_nan"] = n_e_half_with_nan

# Plot electron density vs time.
fig, ax = plt.subplots()
for file_name, data in all_data.items():
    label = file_name.replace("_", " ")
    n_e_with_nan = data["n_e_with_nan"]
    ax.plot(
        times_plasma * 1e9,
        n_e_with_nan * 1e-6,
        label=plot_options[file_name]["label"],
        color=plot_options[file_name]["color"],
        linestyle=plot_options[file_name]["linestyle"],
    )
ax.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
ax.set_ylabel(r"$\mathregular{n_e \, [cm^{-3}]}$")
ax.set_title("Electron density in the plasma")
ax.set_yscale("log")
ax.legend(fontsize=20)
ax.set_xlim(0, 200)
ax.set_ylim(1e14, 1e17)
plt.show()


# %%
# Plot the electron density vs time.
# ----------------------------------

set_mpl_style(nb_columns=2)

# Create the figure.
fig = plt.figure(layout="none", figsize=(16, 20))
gs = fig.add_gridspec(
    nrows=5,
    ncols=2,
    hspace=0.15,
    wspace=0.07,
)
ax1_E = fig.add_subplot(gs[0, 0])
ax2_E = fig.add_subplot(gs[0, 1])
ax1_mu = fig.add_subplot(gs[1, 0])
ax2_mu = fig.add_subplot(gs[1, 1])
ax1_R = fig.add_subplot(gs[2, 0])
ax2_R = fig.add_subplot(gs[2, 1])
ax1_ne = fig.add_subplot(gs[-2:, 0])
ax2_ne = fig.add_subplot(gs[-2:, 1])

# Define annotation kwargs for later use.
kwargs_annotation = dict(
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=20,
    zorder=4,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

# Load experimental data of electron density from OES measurements.
oes_data = np.genfromtxt(
    get_path_to_data("Minesi2022", "fig7_ne_vs_time.dat"),
    delimiter="\t",
    skip_header=3,
    skip_footer=3,
)
# .. Time.
time_ns = oes_data[:, 0]  # [ns]
time_ns += L / c * 1e9  # Shift time to account for propagation delay.
time_ns += 5  # Shift time to match OES measurements with electrical ones.
# .. Electron densities.
ne_4_75mm = oes_data[:, 1]  # [cm^-3]
ne_4_25mm = oes_data[:, 2]  # [cm^-3]
ne_2_25mm = oes_data[:, 3]  # [cm^-3]
ne_0_75mm = oes_data[:, 4]  # [cm^-3]
ne_0_25mm = oes_data[:, 5]  # [cm^-3]
# Uncertainties on electron densities.
u_ne_4_75mm = oes_data[:, 6]  # [cm^-3]
u_ne_4_25mm = oes_data[:, 7]  # [cm^-3]
u_ne_2_25mm = oes_data[:, 8]  # [cm^-3]
u_ne_0_75mm = oes_data[:, 9]  # [cm^-3]
u_ne_0_25mm = oes_data[:, 10]  # [cm^-3]

# Plot experimental data from OES measurements, using the same markers/colors
# as in [Minesi2022]_ Figure 7.
ne_s = [ne_4_75mm, ne_4_25mm, ne_2_25mm, ne_0_75mm, ne_0_25mm]
u_ne_s = [u_ne_4_75mm, u_ne_4_25mm, u_ne_2_25mm, u_ne_0_75mm, u_ne_0_25mm]
labels = ["4.75 mm", "4.25 mm", "2.25 mm", "0.75 mm", "0.25 mm"]
markers = ["D", "v", "^", "o", "s"]
colors = ["m", "g", "b", "r", "k"]
xy_labels = [(105, 1.2e16), (105, 3e15), (105, 1e15), (105, 7e15), (105, 2e16)]
for ne, u_ne, label, marker, color, xy in zip(
    ne_s, u_ne_s, labels, markers, colors, xy_labels
):
    for ax in (ax1_ne, ax2_ne):
        # Plot experimental electron density without error bars.
        ax.scatter(
            time_ns,
            ne,
            s=100,
            marker=marker,
            color=color,
            zorder=3,
        )
        # Plot experimental electron density with error bars, every 4 points.
        ax.errorbar(
            time_ns[::4],
            ne[::4],
            yerr=u_ne[::4],
            fmt=marker,
            markersize=10,
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=5,  # Size of the error bar caps
            zorder=3,
        )
    # .. Annotate the electron density plot (experimental).
    ax2_ne.text(
        *xy,
        s=label,
        color=color,
        **kwargs_annotation,
    )


# Plot numerical results of electron density.
for ax in (ax1_ne, ax2_ne):
    for file_name, data in all_data.items():
        label = file_name.replace("_", " ")

        # Plot electron density (normal radius).
        n_e_with_nan = data["n_e_with_nan"]
        ax.plot(
            times_plasma * 1e9,
            n_e_with_nan * 1e-6,
            zorder=2,
            label=plot_options[file_name]["label"],
            color=plot_options[file_name]["color"],
            linestyle=plot_options[file_name]["linestyle"],
        )

        # Also plot the results with half the discharge radius.
        n_e_half_with_nan = data["n_e_half_with_nan"]
        ax.plot(
            times_plasma * 1e9,
            n_e_half_with_nan * 1e-6,
            label=plot_options[file_name]["label"] + " (r/2)",
            color=plot_options[file_name]["color"],
            linestyle="-.",
            linewidth=6,
        )

# Plot plasma resistance on the resistance figure.
for ax in (ax1_R, ax2_R):
    ax.plot(
        times_expe * 1e9,
        expected_R_p * 1e0,
        label="Extracted plasma resistance",
        color="k",
        ls="--",
        lw=3,
    )
    ax.plot(
        times_expe_with_nan * 1e9,
        expected_R_p_with_nan,
        label="Extracted plasma resistance",
        color="black",
    )

# Plot mobility on the mobility figure.
for ax in (ax1_mu, ax2_mu):
    for file_name, data in all_data.items():
        label = file_name.replace("_", " ")

        mu_interpolated = data["mu_interpolated"]
        ax.plot(
            times_plasma * 1e9,
            mu_interpolated,
            label=plot_options[file_name]["label"],
            color=plot_options[file_name]["color"],
            linestyle=plot_options[file_name]["linestyle"],
        )

# Plot electric field on the electric field figure.
for ax in (ax1_E, ax2_E):
    ax.plot(
        times_plasma * 1e9,
        E_N_3000K_Td,
        label="E/N (3000 K)",
        color="orange",
    )
    ax.plot(
        times_plasma * 1e9,
        E_N_2000K_Td,
        label="E/N (2000 K)",
        color="magenta",
    )

# Plot settings.
for ax1, ax2 in [
    (ax1_ne, ax2_ne),
    (ax1_R, ax2_R),
    (ax1_mu, ax2_mu),
    (ax1_E, ax2_E),
]:
    # .. Hide the spines between ax and ax2.
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    # .. Add diagonal lines to indicate the break in the x-axis.
    d = 1.5  # Proportion of vertical to horizontal extent of the slanted line.
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=25,
        linestyle="none",
        color="k",
        mec="k",
        mew=3,
        clip_on=False,
    )
    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)  # type: ignore
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)  # type: ignore

    # .. Change x-limits.
    ax1.set_xlim(40, 60)
    ax2.set_xlim(105, 140)

    # .. Remove x-ticks for all plots except the bottom ones.
    ax1.set_xticks([40, 45, 50, 55])
    ax2.set_xticks([110, 120, 130])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    # .. Place the y-axis labels in the middle of the plots.
    ax1.yaxis.set_label_coords(-0.12, 0.5)

# .. Set x-ticks, only for the bottom plots.
ax1_ne.set_xticks([40, 45, 50, 55])
ax1_ne.set_xticklabels(["40", "45", "50", "55"])
ax2_ne.set_xticks([110, 120, 130])
ax2_ne.set_xticklabels(["110", "120", "130"])

# .. Change y-scale and y-limits
for ax in (ax1_ne, ax2_ne):
    ax.set_yscale("log")
    ax.set_ylim(1e14, 3e16)
for ax in (ax1_R, ax2_R):
    ax.set_ylim(-100, 1000)
for ax in (ax1_mu, ax2_mu):
    ax.set_ylim(0.1, 0.5)
for ax in (ax1_E, ax2_E):
    ax.set_ylim(-50, 600)

# .. Add labels.
ax1_ne.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax1_ne.set_ylabel(r"$\mathregular{n_e \, [cm^{-3}]}$")
ax1_R.set_ylabel(r"$\mathregular{R_{plasma} \, [\Omega]}$")
ax1_mu.set_ylabel(r"$\mathregular{\mu \, [m^2/(V \cdot s)]}$")
ax1_E.set_ylabel(r"$\mathregular{|E_{plasma}|/n \, [Td]}$")

# .. Change position of x-axis label.
ax1_ne.xaxis.set_label_coords(1.05, -0.05)
ax1_R.xaxis.set_label_coords(1.05, -0.05)

# .. Annotate the electron density plot (numerical).
ax1_ne.text(
    45,
    4e14,
    "Simulation\n(2000 K)",
    color="magenta",
    **kwargs_annotation,  # type: ignore
)
ax1_ne.text(
    50,
    8e14,
    "Simulation\n(3000 K)",
    color="orange",
    **kwargs_annotation,  # type: ignore
)

ax1_ne.text(
    43,
    1e16,
    "Simulation\n(2000 K, r/2)",
    color="magenta",
    **kwargs_annotation,  # type: ignore
)
ax1_ne.text(
    47,
    2e16,
    "Simulation\n(3000 K, r/2)",
    color="orange",
    **kwargs_annotation,  # type: ignore
)
# Create arrows for the (r/2) annotations.
ax1_ne.annotate(
    "",
    xy=(46, 3e15),
    xytext=(43, 8e15),
    arrowprops=dict(color="magenta", arrowstyle="->", lw=3),
)
ax1_ne.annotate(
    "",
    xy=(48.4, 3e15),
    xytext=(48.4, 2e16),
    arrowprops=dict(color="orange", arrowstyle="->", lw=3),
)

# .. Annotate the electric field plot.
ax1_E.text(45, 300, "2000 K", color="magenta", **kwargs_annotation)  # type: ignore
ax1_E.text(50, 350, "3000 K", color="orange", **kwargs_annotation)  # type: ignore

# .. Annotate the mobility plot.
ax1_mu.text(45, 0.18, "2000 K", color="magenta", **kwargs_annotation)  # type: ignore
ax1_mu.text(50, 0.3, "3000 K", color="orange", **kwargs_annotation)  # type: ignore

plt.show()

# %%
# Save the figure.
# ----------------

# Export the image to a .svg file, in the figures folder.
fig.savefig(
    get_path_to_data(
        "article_figures",
        "Minesi2022_comparison_electron_density.svg",
        force_return=True,
    ),
)

# %%
