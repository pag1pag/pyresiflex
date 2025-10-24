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
times_expe_with_nan = expe.times_corrected_with_nan
expected_R_p = expe.Rp_corrected
expected_R_p_with_nan = expe.Rp_corrected_with_nan


# %%
# Get the electron-neutral collision frequency.
# ---------------------------------------------

times_plasma, nu_m_3000K, nu_m_2000K = np.loadtxt(
    get_path_to_data(
        "cross_sections",
        "Bolsig",
        "output",
        "momentum_transfer_frequency_vs_time.csv",
    ),
    delimiter=",",
    skiprows=1,
    unpack=True,
)
# plot the momentum transfer frequency vs time.
fig, ax = plt.subplots()
ax.plot(times_plasma * 1e9, nu_m_3000K, label="3000 K")
ax.plot(times_plasma * 1e9, nu_m_2000K, label="2000 K")
ax.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
ax.set_ylabel(r"$\mathregular{\nu_m \, [Hz]}$")
ax.set_title("Momentum transfer frequency in the plasma")
ax.legend()
plt.show()

# %%
# Interpolate the momentum transfer frequency to match experimental times,
# taking care of the mismatch in time windows.

nu_en_3000K = np.interp(
    times_expe,
    times_plasma + L / c,
    nu_m_3000K,
    left=0,
    right=0,
)
nu_en_2000K = np.interp(
    times_expe,
    times_plasma + L / c,
    nu_m_2000K,
    left=0,
    right=0,
)
nu_en_3000K_with_nan = np.interp(
    times_expe_with_nan,
    times_plasma + L / c,
    nu_m_3000K,
    left=np.nan,
    right=np.nan,
)
nu_en_2000K_with_nan = np.interp(
    times_expe_with_nan,
    times_plasma + L / c,
    nu_m_2000K,
    left=np.nan,
    right=np.nan,
)
# plot the momentum transfer frequency vs time.
fig, ax = plt.subplots()
ax.plot(
    times_expe * 1e9, nu_en_3000K, label="3000 K", ls="--", lw=2, color="k"
)
ax.plot(
    times_expe * 1e9, nu_en_2000K, label="2000 K", ls="--", lw=2, color="r"
)
ax.plot(
    times_expe_with_nan * 1e9,
    nu_en_3000K_with_nan,
    label="3000 K",
    ls="-",
    lw=4,
    color="k",
)
ax.plot(
    times_expe_with_nan * 1e9,
    nu_en_2000K_with_nan,
    label="2000 K",
    ls="-",
    lw=4,
    color="r",
)
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{\nu_m \, [Hz]}$")
ax.set_title("Momentum transfer frequency in the plasma")
ax.legend()
ax.set_xlim(0, 150)
plt.show()

# %%
# Get back to electron density.
# -----------------------------------

# .. Geometry parameters
#  Gap distance.
d_gap = 5e-3  # [m]
# Discharge radius.
r_discharge = 0.6e-3  # [m]
# Discharge section.
S_discharge = np.pi * r_discharge**2  # [m^2]


# Electron density.
def n_e_from_R_p(plasma_resistance, electron_neutral_collision_frequency):
    return (
        u.m_e
        / u.e**2
        * electron_neutral_collision_frequency
        * 1
        / (plasma_resistance * S_discharge / d_gap)
    )  # [m^-3]


n_e_3000K = n_e_from_R_p(expected_R_p, nu_en_3000K)  # [m^-3]
n_e_3000K_with_nan = n_e_from_R_p(
    expected_R_p_with_nan, nu_en_3000K_with_nan
)  # [m^-3]
n_e_2000K = n_e_from_R_p(expected_R_p, nu_en_2000K)  # [m^-3]
n_e_2000K_with_nan = n_e_from_R_p(
    expected_R_p_with_nan, nu_en_2000K_with_nan
)  # [m^-3]

## %%
## Plot the electron density vs time.
## ----------------------------------

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
    mask_with_nan = (times_expe_with_nan * 1e9 >= t_start) & (
        times_expe_with_nan * 1e9 <= t_end
    )
    for ax in (ax1, ax2):
        # Plot results for 3000 K.
        ax.plot(
            times_expe[mask] * 1e9,
            n_e_3000K[mask] * 1e-6,  # Convert to cm^-3.
            color="magenta",
            ls="--",
            lw=2,
        )
        ax.plot(
            times_expe_with_nan * 1e9,
            n_e_3000K_with_nan * 1e-6,  # Convert to cm^-3.
            color="magenta",
            ls="-",
        )
        # Plot results for 2000 K.
        ax.plot(
            times_expe[mask] * 1e9,
            n_e_2000K[mask] * 1e-6,  # Convert to cm^-3.
            color="orange",
            lw=2,
            ls="--",
        )
        ax.plot(
            times_expe_with_nan * 1e9,
            n_e_2000K_with_nan * 1e-6,  # Convert to cm^-3.
            color="orange",
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
