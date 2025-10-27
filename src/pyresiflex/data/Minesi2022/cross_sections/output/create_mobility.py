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

from pyresiflex.misc.plot import plot_voltage_current, set_mpl_style
from pyresiflex.misc.units import k_b
from pyresiflex.misc.utils import get_path_to_data

set_mpl_style(nb_columns=2)


# %%
# Load [Minesi2022]_ experimental data of anode configuration.
# -------------------------------------------------------------

# Load the raw data from Figure 16 of [Minesi2022]_.
file = get_path_to_data(
    "Minesi2022",
    "fig3_anodeConfiguration.csv",
)
data = np.loadtxt(file, skiprows=3, delimiter=";")
times_raw_anode = data[:, 0]  # [s]
voltages_raw_anode = data[:, 1] * 1e3  # [V]
currents_raw_anode = data[:, 2]  # [A]


# Plot the raw data.
fig, ax_v, ax_i = plot_voltage_current(
    voltage_time=times_raw_anode,
    voltage_value=voltages_raw_anode,
    current_time=times_raw_anode,
    current_value=currents_raw_anode,
)
ax_v.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
ax_v.set_ylim(-4, 8)
ax_i.set_ylim(-25, 50)

plt.show()

# Define the zero at the first time the voltage reaches `threshold_voltage`.
threshold_voltage = 200  # [V]
idx_first = np.where(np.abs(voltages_raw_anode) > threshold_voltage)[0][0]
times_raw_anode = times_raw_anode - times_raw_anode[idx_first]

# Define a time window to analyze.
lower_time_window = -20e-9  # [s]
upper_time_window = 200e-9  # [s]

# Limit the time window to [lower_time_window, upper_time_window]
idx_min_wanted_time = np.where(times_raw_anode > lower_time_window)[0][0]
idx_max_wanted_time = np.where(times_raw_anode > upper_time_window)[0][0]

# Limit the time, voltages and currents to the wanted period.
times_expe_anode = times_raw_anode[idx_min_wanted_time:idx_max_wanted_time]
voltages_expe_anode = voltages_raw_anode[
    idx_min_wanted_time:idx_max_wanted_time
]
currents_expe_anode = currents_raw_anode[
    idx_min_wanted_time:idx_max_wanted_time
]

# Compute the energy from the voltage and current.
energies_expe_anode = np.zeros_like(times_expe_anode)  # [J]
for i in range(len(times_expe_anode)):
    energies_expe_anode[i] = np.trapezoid(
        voltages_expe_anode[:i] * currents_expe_anode[:i], times_expe_anode[:i]
    )


# Plot the preprocessed data.
fig, ax_v, ax_i = plot_voltage_current(
    voltage_time=times_expe_anode,
    voltage_value=voltages_expe_anode,
    current_time=times_expe_anode,
    current_value=currents_expe_anode,
)
ax_v.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
ax_v.set_ylim(-4, 8)
ax_i.set_ylim(-25, 50)
ax_i.set_yticks([-25, -15, -5, 5, 15, 25])

plt.show()

# %%
# Extract the plasma voltage, between 0 and 50 ns.
# ---------------------------------------------------

mask_time = (times_expe_anode >= 0) & (times_expe_anode <= 200 - 9)
times_plasma = times_expe_anode[mask_time]
plasma_voltage = voltages_expe_anode[mask_time]

fig, ax = plt.subplots()
ax.plot(times_plasma * 1e9, plasma_voltage * 1e-3)
ax.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
ax.set_ylabel(r"$\mathregular{V_{plasma} \, [kV]}$")
ax.set_title("Extracted plasma voltage")
ax.set_ylim(-1, 7)
plt.show()

gap = 5e-3  # [m]
plasma_electric_field = np.abs(plasma_voltage) / gap  # [V/m]
n_gas = 2.5e24  # [m^-3], neutral density at 1 atm and 3000 K.
reduced_electric_field = plasma_electric_field / n_gas  # [V m^2]

reduced_electric_field_Td = reduced_electric_field * 1e21  # [Td]

fig, ax = plt.subplots()
ax.plot(times_plasma * 1e9, reduced_electric_field_Td)
ax.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
ax.set_ylabel(r"$\mathregular{E/N \, [Td]}$")
ax.set_title("Reduced electric field in the plasma")
# ax.set_ylim(0, 300)
plt.show()


# %%
# Get momentum transfer frequency from Bolsig+.
# ---------------------------------------------

output_folders_3000K = get_path_to_data(
    "cross_sections",
    "Bolsig",
    "output",
    "3000K_no_ee_no_ei",
    "extracted_output.csv",
)
output_folders_2000K = get_path_to_data(
    "cross_sections",
    "Bolsig",
    "output",
    "2000K_no_ee_no_ei",
    "extracted_output.csv",
)

data = np.loadtxt(output_folders_3000K, delimiter=",", skiprows=1)
E_N_3000K = data[:, 0]
mu_N_m_3000K = data[:, 3]

data = np.loadtxt(output_folders_2000K, delimiter=",", skiprows=1)
E_N_2000K = data[:, 0]
mu_N_m_2000K = data[:, 3]

fig_E, ax_E = plt.subplots()
ax_E.plot(E_N_3000K, mu_N_m_3000K, label="3000 K")
ax_E.plot(E_N_2000K, mu_N_m_2000K, label="2000 K")
ax_E.set_title("Mobility * N from Bolsig+")
ax_E.set_xlabel(r"$\mathregular{E/N \, [Td]}$")
ax_E.set_ylabel(r"$\mathregular{\mu_N \, [1/(m*V*s)]}$")
ax_E.set_xscale("log")
ax_E.legend()

# %%
# Plot the momentum transfer frequency vs time.
# ---------------------------------------------

# First, interpolate the momentum transfer frequency vs reduced electric field.
mu_N_m_interpolated_3000K = np.interp(
    reduced_electric_field_Td, E_N_3000K, mu_N_m_3000K, left=0, right=0
)
mu_N_m_interpolated_2000K = np.interp(
    reduced_electric_field_Td, E_N_2000K, mu_N_m_2000K, left=0, right=0
)
# Then, plot it vs time.
# fig, ax = plt.subplots()
# ax.plot(times_plasma * 1e9, nu_N_m_interpolated)
# ax.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
# ax.set_ylabel(r"$\mathregular{\nu_m/N \, [m^3/s]}$")
# ax.set_title("Reduced momentum transfer frequency in the plasma")
# plt.show()

P = 1.01315e5  # [Pa]
T = 3000  # [K]
n_gas = P / (k_b * T)  # [m^-3]
mu_m_3000K = mu_N_m_interpolated_3000K / n_gas  # [s^-1]
T = 2000  # [K]
n_gas = P / (k_b * T)  # [m^-3]
mu_m_2000K = mu_N_m_interpolated_2000K / n_gas  # [s^-1]

fig, ax = plt.subplots()
ax.plot(times_plasma * 1e9, mu_m_3000K, label="3000 K")
ax.plot(times_plasma * 1e9, mu_m_2000K, label="2000 K")
ax.set_xlabel(r"$\mathregular{t - \frac{L}{c} \, [ns]}$")
ax.set_ylabel(r"$\mathregular{\mu_m \, [m^2/(V*s)]}$")
ax.set_title("Mobility in the plasma")
ax.legend()
plt.show()

# %%
# Save the momentum transfer frequency vs time.
np.savetxt(
    "mobility_vs_time.csv",
    np.column_stack((times_plasma, mu_m_3000K, mu_m_2000K)),
    delimiter=",",
    header="time [s], "
    "mobility [m^2/(V*s)] at 3000 K, "
    "mobility [m^2/(V*s)] at 2000 K",
)


# %%
