r"""
Determination of cable properties.
==================================

This example shows how to determine the properties of a coaxial cable
from experimental data, and compare them to the datasheet values.
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

import pyresiflex.misc.units as u
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
lower_time_window = -10e-9  # [s]
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
ax_v.set_ylim(-4, 4)
ax_i.set_ylim(-60, 60)
ax_i.set_yticks([-60, -45, -30, -15, 0, 15, 30, 45, 60])

plt.show()

# %%
# Determine the time delay between the incident and reflected wave.
# -----------------------------------------------------------------

# Find the points where the voltage reaches 2 kV.
t_2kV = times_expe[np.where(np.abs(voltages_expe - 2000) < 100)[0]]
print(f"Times when voltage reaches 2 kV: {t_2kV * 1e9} ns")
# Time difference between detection of 2 kV, for the first rising edge.
delta_t = t_2kV[2] - t_2kV[0]  # [s]
print(
    "Time difference between the incident"
    f" and reflected wave: {delta_t * 1e9} ns"
)

fig, ax_v, ax_i = plot_voltage_current(
    voltage_time=times_expe,
    voltage_value=voltages_expe,
    current_time=times_expe,
    current_value=currents_expe,
)
ax_v.set_ylim(-4, 4)
ax_i.set_ylim(-60, 60)
ax_i.set_yticks([-60, -45, -30, -15, 0, 15, 30, 45, 60])

ax_v.axvline(t_2kV[0] * 1e9, color="blue", linestyle="--", lw=3)
ax_v.axvline(t_2kV[2] * 1e9, color="blue", linestyle="--", lw=3)
ax_v.text(
    t_2kV[0] * 1e9,
    -3,
    "Incident wave (2 kV)",
    color="blue",
    ha="center",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
)
ax_v.text(
    t_2kV[2] * 1e9,
    3,
    "Reflected wave (2 kV)",
    color="blue",
    ha="center",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
)

plt.show()

# %%
# Determine the cable properties from the datasheet.
# --------------------------------------------------
#
# We will use the datasheet to get the celerity and characteristic impedance.
# We will then use the time delay to get the length of the cable.


# In [Minesi2022]_, the cable used was a AlphaWire 9011A RG11A/U.
# The datasheet of this cable is available in the data folder.
# The cable has the following properties:
c_datasheet = 0.66 * u.c  # [m/s]
Z_c_datasheet = 75  # [Ohm]


# We need to make an assumption on the position of the probes.
# Here, we assume that the probes are located at the same position,
# at the middle of the cable.
# x = L / 2
#
# Then, the wave velocity is given by:
#   c = 2 * (L - x) / delta_t
#     = L / delta_t
# Thus, the length of the cable is:
L_from_datasheet = c_datasheet * delta_t  # [m]


# %%
# Determine the cable properties from the experimental data.
# ----------------------------------------------------------
#
# [Minesi2022]_ provides the cable length used, roughly 6 m.
# However, there were two cables of 3 m connected in series, via a box,
# where probes are connected. This box adds some additional length,
# roughly 0.2 m.
# Thus, we will assume that the length of the cable is:
L_experimental = 6.2  # [m]
# [Minesi2022]_ also provides the probe positions, at the middle of the cable.
x_experimental = L_experimental / 2  # [m]

# Then, the wave velocity is given by:
c_experimental = 2 * (L_experimental - x_experimental) / delta_t

# The characteristic impedance can be determined from the maximum voltage
# and current of the incident wave:
Z_c = np.max(voltages_expe) / np.max(currents_expe)

# %%
# Compare the results.
# --------------------

print("Cable properties from datasheet:")
print(f"  Length: {L_from_datasheet:.2f} m")
print(f"  Celerity: {c_datasheet / u.c:.2f} c = {c_datasheet:.2e} m/s")
print(f"  Characteristic impedance: {Z_c_datasheet:.2f} Ohms")
print("")
print("Cable properties from experimental data:")
print(f"  Length: {L_experimental:.2f} m")
print(f"  Celerity: {c_experimental / u.c:.2f} c = {c_experimental:.2e} m/s")
print(f"  Characteristic impedance: {Z_c:.2f} Ohms")


# %%
# Determining if the cable can be considered perfect or not.
# ----------------------------------------------------------

# feet to meters conversion: 1 ft = 0.3048 m
feet_to_m = 0.3048  # [m/ft]

print("Cable properties from datasheet:")
# Inner conductor resistance per unit length:
R_ohm_feet = 6.3 / 1000  # [Ohm/ft]
R_ohm_m = R_ohm_feet / feet_to_m  # [Ohm/m]
print(f"Cable resistance: {R_ohm_m * 1e3:.1f} Ohm/km")

# Velocity of propagation:
c = 0.66 * u.c  # [m/s] (from datasheet)

# Ground capacitance per unit length:
C_F_per_feet = 20.5e-12  # [F/ft]
C_F_per_m = C_F_per_feet / feet_to_m  # [F/m]
print(f"Cable capacitance: {C_F_per_m * 1e12:.1f} pF/m")

# Inductance per unit length:
L_H_per_m = 1 / (c**2 * C_F_per_m)  # [H/m]
print(f"Cable inductance: {L_H_per_m * 1e9:.1f} nH/m")


# Cable attenuation:
alpha_dB_per_ft = [
    5.2 / 100,  # [dB/ft] at 400 MHz
    9.4 / 100,  # [dB/ft] at 1 GHz
]
alpha_dB_per_m = [a / feet_to_m for a in alpha_dB_per_ft]  # [dB/m]
f_attenuation = [400e6, 1e9]  # [Hz]
print(f"Cable attenuation: {alpha_dB_per_m[0]:.2f} dB/m at 400 MHz")
print(f"Cable attenuation: {alpha_dB_per_m[1]:.2f} dB/m at 1 GHz")
# Create a function to (linearly) interpolate the attenuation at any frequency.
alpha_dB_per_m_func = np.poly1d(
    np.polyfit(f_attenuation, alpha_dB_per_m, deg=1)
)
for f in [0, 100e6, 200e6, 300e6, 400e6, 600e6, 800e6, 1e9]:
    print(f"  at {f / 1e6:.0f} MHz: {alpha_dB_per_m_func(f):.2f} dB/m")


# Trapezoidal pulse parameters:
print("-" * 40)
print("Trapezoidal pulse parameters:")
t_rising = 2e-9  # [s]
f_rising = 0.35 / t_rising  # [Hz]
t_flat = 10e-9  # [s]
f_flat = 1 / t_flat  # [Hz]
print(f"Rising edge frequency: {f_rising / 1e6:.1f} MHz")
print(f"Flat top frequency: {f_flat / 1e6:.1f} MHz")

# Attenuation at the rising edge frequency:
alpha_rising = alpha_dB_per_m_func(f_rising)  # [dB/m]
# Attenuation at the flat top frequency:
alpha_flat = alpha_dB_per_m_func(f_flat)  # [dB/m]
print(f"Cable attenuation at rising edge frequency: {alpha_rising:.2f} dB/m")
print(f"Cable attenuation at flat top frequency: {alpha_flat:.2f} dB/m")

# Attenuation over the whole cable length:
print("-" * 40)
for L in [10, 24]:  # [m]
    print(f"Attenuation of voltage over {L} m of cable:")
    total_attenuation = 10 ** (-alpha_rising * L / 20)
    print(f"  at rising edge frequency: {total_attenuation:.3f} (-)")
    total_attenuation = 10 ** (-alpha_flat * L / 20)
    print(f"  at flat top frequency: {total_attenuation:.3f} (-)")

# %%
