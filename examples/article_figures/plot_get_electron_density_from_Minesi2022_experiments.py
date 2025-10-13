r"""
Compare electron density results between electrical and optical methods.
========================================================================

In this example, the plasma is assumed to be a time-varying resistance.

Experimental data of [Minesi2022]_ are used to extract the plasma resistance.
For the remote configuration case, voltage and current signals are measured at
the middle of a cable of length :math:`L \approx 6 \, \mathrm{m}`.

This example shows how to extract the resistance of a plasma load from a
measured voltage and measured current, at the same position.
Then, electron density is estimated from the plasma resistance, and compared
with values obtained via Optical Emission Spectroscopy (OES).
"""  # noqa: D205

# This sets the fifth figure as the thumbnail for the example gallery.
# sphinx_gallery_thumbnail_number = 5
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

import pyresiflex.misc.units as u
from pyresiflex.experiment.purely_resistive_experiment import (
    PurelyResistiveExperiment,
)
from pyresiflex.misc.plot import plot_voltage_current, set_mpl_style
from pyresiflex.misc.utils import get_path_to_data

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
ax_v.set_ylim(-4, 4)
ax_i.set_ylim(-60, 60)
ax_i.set_yticks([-60, -45, -30, -15, 0, 15, 30, 45, 60])

plt.show()

# %%
# Transmission line parameters
# ----------------------------

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

# Voltage and current measured, corrected for propagation delay.
fig, ax_v, ax_i = expe.plot_voltage_and_current()
ax_v.set_xlim(0, 150)
ax_v.set_ylim(-4, 4)
ax_i.set_ylim(-60, 60)
ax_i.set_yticks([-60, -45, -30, -15, 0, 15, 30, 45, 60])

expe.compute_plasma_resistance_from_vmes_and_imes(
    times_expe, threshold=400, channel_formation_time=42e-9
)

fig, ax = expe.plot_resistance(times=times_expe)
ax.set_xlim(0, 150)
ax.set_ylim(-100, 1000)
plt.show()

times_expe = expe.times_corrected
expected_R_p = expe.Rp_corrected


# %%
# Get back to electron density.
# -----------------------------------

# Gas pressure.
P_gas = 1e5  # [Pa]
# Gas temperature.
T_gas = 3000  # [K]
# Gas density, assuming ideal gas and constant density.
n_gas = P_gas / (u.k_b * T_gas)  # [m^-3]

# Average momentum-transfer cross-section for electron-neutral collision.
# Here, it is assumed that there is only nitrogen in the gas.
# Experimental data from LXCat.
Q_en = 1e-19  # m^2

# Electron temperature.
T_e = 4e4  # [K]
# Electron thermal velocity.
v_th_e = np.sqrt(8 * u.k_b * T_e / (np.pi * u.m_e))  # [m/s]

# Electron-neutral collision frequency.
nu_en = n_gas * Q_en * v_th_e  # [Hz]

print(f"Electron-neutral collision frequency: {nu_en:.2e} Hz")

# Electrical conductivity from electron-ion collisions.
C = 7.5e-3
log_lambda = 10  # Coulomb logarithm.
sigma_ei = C * T_e**1.5 / log_lambda  # [S/m]

# .. Geometry parameters
#  Gap distance.
d_gap = 5e-3  # [m]
# Discharge radius.
r_discharge = 0.6e-3  # [m]
# Discharge section.
S_discharge = np.pi * r_discharge**2  # [m^2]
# Discharge radius corrected.
r_discharge_corrected = r_discharge / 2  # [m]
# Discharge section corrected.
S_discharge_corrected = np.pi * r_discharge_corrected**2  # [m^2]

# Electron density.
n_e = (
    u.m_e
    / u.e**2
    * nu_en
    * 1
    / ((expected_R_p * S_discharge / d_gap) - 1 / sigma_ei)
)  # [m^-3]
# Electron density corrected (surface has changed).
n_e_corrected = (
    u.m_e
    / u.e**2
    * nu_en
    * 1
    / ((expected_R_p * S_discharge_corrected / d_gap) - 1 / sigma_ei)
)  # [m^-3]
# Remove negative values, which are non-physical.
n_e_corrected[n_e_corrected < 0] = np.nan


# %%
# Plot the electron density vs time.
# ----------------------------------

# Create the figure.
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, facecolor="w")

# Load experimental data of electron density from OES measurements.
time_ns, ne_4_75mm, ne_4_25mm, ne_2_25mm, ne_0_75mm, ne_0_25mm = np.genfromtxt(
    get_path_to_data("Minesi2022", "fig7_ne_vs_time.dat"),
    delimiter="\t",
    skip_header=3,
    skip_footer=3,
    unpack=True,
    usecols=(0, 1, 2, 3, 4, 5),
)
time_ns += L / c * 1e9  # Shift time to account for propagation delay.
time_ns += 5  # Shift time to match OES measurements with electrical ones.

# Plot experimental data from OES measurements.
ne_s = [ne_4_75mm, ne_4_25mm, ne_2_25mm, ne_0_75mm, ne_0_25mm]
labels = ["4.75 mm", "4.25 mm", "2.25 mm", "0.75 mm", "0.25 mm"]
markers = ["D", "v", "^", "o", "s"]
colors = ["m", "g", "b", "r", "k"]
for ne, label, marker, color in zip(ne_s, labels, markers, colors):
    for ax in (ax1, ax2):
        ax.scatter(
            time_ns,
            ne,
            s=150,
            marker=marker,
            color=color,
            label=label,
            zorder=3,
        )


# Plot numerical results, on different time interval,
# to avoid interpolation artifacts.
for t_start, t_end in [(40, 70), (90, 120), (120, 140)]:
    mask = (times_expe * 1e9 >= t_start) & (times_expe * 1e9 <= t_end)
    for ax in (ax1, ax2):
        ax.plot(
            times_expe[mask] * 1e9,
            n_e[mask] * 1e-6,  # Convert to cm^-3.
            color="magenta",
            ls="-",
        )

        ax.plot(
            times_expe[mask] * 1e9,
            n_e_corrected[mask] * 1e-6,  # Convert to cm^-3.
            color="k",
            lw=4,
            ls="-",
        )


# Plot settings.

# .. Change x-limits and y-limits.
ax1.set_xlim(40, 60)
ax1.set_yscale("log")
ax1.set_ylim(1e14, 3e16)
ax2.set_xlim(105, 140)

# Hide the spines between ax and ax2.
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()

# Add diagonal lines to indicate the break in the x-axis.
d = 2  # Proportion of vertical to horizontal extent of the slanted line.
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=40,
    linestyle="none",
    color="k",
    mec="k",
    mew=3,
    clip_on=False,
)
ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

# Set x-ticks.
ax1.set_xticks([40, 45, 50, 55])
ax2.set_xticks([110, 120, 130, 140])

# Add labels.
ax1.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax1.set_ylabel(r"$\mathregular{n_e \, [cm^{-3}]}$")
# Change position of x-axis label.
ax1.xaxis.set_label_coords(1.05, -0.05)

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
