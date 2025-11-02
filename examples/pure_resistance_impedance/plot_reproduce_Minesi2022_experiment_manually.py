r"""
(Manually) Reproducing [Minesi2022]_ experiment.
================================================

In this example, the plasma is a time-varying resistance.

Experimental data of [Minesi2022]_ are used to extract the plasma resistance.
For the remote configuration case, voltage and current signals are measured at
the middle of a cable of length :math:`L \approx 6 \, \mathrm{m}`.


To run the simulations, various parameters are needed:

- for the transmission line:

  - Length :math:`L \approx 6 \, \mathrm{m}`
  - Wave velocity :math:`c \approx 1.8 \times 10^8 \, \mathrm{m/s}`
  - Characteristic impedance :math:`Z_c \approx 75 \, \Omega`

- for the generator:

  - Generator voltage equals to the first incident voltage multiplied by
    an attenuation factor :math:`\alpha_g`
  - Internal resistance of :math:`R_g \approx 10 \, \Omega`

- for the load:

  - Plasma resistance is a time-varying resistance, whose values are
    manually chosen to reproduce, a posteriori, the experimental data.

"""  # noqa: D205

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
from pyresiflex.generator.generator_real_impedance import (
    FromMeasurementGenerator,
)
from pyresiflex.load.time_varying_resistance import PlasmaResistanceInterpolate
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
R_g = 1  # [Ohm]
# Attenuation coefficient.
alpha_g = 1 / (1 + R_g / Z_c)  # [-]
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
fig, ax = plt.subplots()
# fig.suptitle("Voltage and current signals from Minesi2022")
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
    color="blue",
)
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{V \, [kV]}$")
ax.set_xlim(0, 50)
ax.set_ylim(-0.1, 4.0)
ax.legend()
plt.show()


# %%
# Load parameters.
# ----------------

plasma_load = PlasmaResistanceInterpolate(
    t_array=np.array([0, 13, 22, 30, 50, 80]) * 1e-9 + L / c,
    R_array=np.array([1000, 100, 1, 5, 50, 200]),
)

# %%
# Solution object.
# ----------------

solution = PurelyResistiveSolution(
    generator=generator,
    load=plasma_load,
    cable=cable,
)


# %%
# Plot the plasma resistance vs time.
# -----------------------------------

fig, ax = plt.subplots()

# Compute the plasma resistance at different times_expe.
R_p = np.array([solution.R_l(t) for t in times_expe])

ax.plot(
    times_expe * 1e9,
    R_p,
    color="k",
    label="Plasma resistance",
)
ax.hlines(
    Z_c,
    xmin=0,
    xmax=200,
    color="r",
    ls="--",
    label="Cable impedance",
)
ax.vlines(
    x=solution.L / solution.c * 1e9,
    ymin=0,
    ymax=1000,
    ls="--",
    label="Time of the reflexion",
)

# ax.set_title("Resistance vs time")
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{R_p \, [\Omega]}$")
ax.set_xlim(0, 200)
ax.legend()
plt.show()


# %%
# Compute voltage and current at a given position.
# ------------------------------------------------

# Time vector for the simulation.
nb_steps = 1000
times = np.linspace(lower_time_window, upper_time_window, nb_steps)  # [s]
# Position of probes for measurement
x = 3.1  # [m]


# Compute the voltage and current at probes' position.
solution.solve(x, times)

voltages = solution.voltage  # [V]
currents = solution.current  # [A]
energies = solution.energy  # [J]
xs = solution.x  # [m]
times = solution.t  # [s]


# %%
# Plot the voltage and current at a given position.
# -------------------------------------------------

# This sets the following figure as the thumbnail for the example gallery.
# sphinx_gallery_thumbnail_number = 5

# Do we want to plot the current and energy?
plot_current = True
plot_energy = True


# Compute the min and max values of the voltage and current,
# to have the same zero level for voltage and current.
max_abs_voltage = np.max(np.abs(voltages)) * 1.1
max_abs_current = np.max(np.abs(currents)) * 1.1

fig, ax_v = plt.subplots()

# .. Set title
suptitle = "Voltage"
if plot_current and plot_energy:
    suptitle += ", current and energy"
elif plot_current:
    suptitle += " and current"
elif plot_energy:
    suptitle += " and energy"
suptitle += f" at x = {x:.2f} m"
# fig.suptitle(suptitle, y=1.01)


# Ohm = r"$\Omega$"
# title = f"Cable: L = {L:.2f} m, $Z_c$ = {Z_c:.1f} {Ohm}, c = {c:.1e} m/s\n"
# ax_v.set_title(title, fontsize=12, loc="left")


# Plot voltage.
plot_line_v = ax_v.plot(
    (times - x / c) * 1e9,
    voltages * 1e-3,
    color="k",
    ls="--",
    label="Voltage (computed)",
)
plot_line_v_measured = ax_v.plot(
    times_expe * 1e9,
    voltages_expe * 1e-3,
    color="k",
    label="Voltage (experimental)",
)
# .. Plot options for voltage.
ax_v.set_xlabel(r"$\mathregular{t - \frac{x_{meas}}{c} \, [ns]}$")
ax_v.set_ylabel(r"$\mathregular{V \, [kV]}$")
# ax_v.set_ylim(-max_abs_voltage * 1e-3, max_abs_voltage * 1e-3)
ax_v.set_ylim(-4, 4)
ax_v.spines["left"].set_color("k")
ax_v.set_xlim(times[0] * 1e9, times[-1] * 1e9)

# Plot current.
if plot_current:
    ax_i = ax_v.twinx()
    plot_line_i = ax_i.plot(
        (times - x / c) * 1e9,
        currents,
        color="r",
        ls="--",
        label="Current (computed)",
    )
    plot_line_i_measured = ax_i.plot(
        times_expe * 1e9,
        currents_expe,
        color="r",
        label="Current (experimental)",
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

# Plot energy.
if plot_energy:
    ax_e = ax_v.twinx()
    plot_line_e = ax_e.plot(
        (times - x / c) * 1e9,
        energies * 1e3,
        color="b",
        ls="--",
        label="Energy",
    )
    plot_line_e_measured = ax_e.plot(
        times_expe * 1e9,
        energies_expe * 1e3,
        color="b",
        label="Energy (experimental)",
    )
    # .. Plot options for energy.
    ax_e.set_ylabel(r"$\mathregular{E \, [mJ]}$", color="b")
    # Move the y-axis of ax_e to the right, by 100 points
    ax_e.spines["right"].set_position(("outward", 100))
    ax_e.grid(visible=False)
    ax_e.set_ylim(0, 2.5)
    # Change color of the right y-axis to blue.
    ax_e.spines["right"].set_color("b")
    # Also change the color of the ticks.
    ax_e.tick_params(axis="y", colors="b")


# .. Display all legends in the same box.
lines = plot_line_v + plot_line_v_measured
if plot_current and plot_line_i:
    lines += plot_line_i + plot_line_i_measured
if plot_energy and plot_line_e:
    lines += plot_line_e + plot_line_e_measured

labels: list[str] = []
for line in lines:
    label = line.get_label()
    if isinstance(label, str):
        labels.append(label)
    else:
        labels.append("Unknown")
# ax_v.legend(lines, labels, loc="lower right")

ax_v.legend(
    handles=plot_line_v + plot_line_v_measured,
    labels=["Computed", "Measured"],
    loc="lower right",
)

plt.show()


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
fig, ax_v = plt.subplots()
# fig.suptitle("Raw data: Voltage and current signals from Minesi2022")
# .. Plot the voltage signal.
ax_v.plot(
    times_raw_anode * 1e9,
    voltages_raw_anode * 1e-3,
    label="Voltage",
    color="black",
)
ax_v.set_ylim(-4, 8)
ax_v.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax_v.set_ylabel(r"$\mathregular{V \, [kV]}$")
# .. Add a second y-axis for the current signal.
ax_i = ax_v.twinx()
ax_i.plot(
    times_raw_anode * 1e9, currents_raw_anode, label="Current", color="red"
)
ax_i.set_ylabel(r"$\mathregular{I \, [A]}$", color="r")
ax_i.set_ylim(-25, 50)
ax_i.grid(visible=False)
ax_i.spines["right"].set_color("r")
ax_i.tick_params(axis="y", color="r", labelcolor="r")

plt.show()

# Preprocess the data.
# Limit the time window to [lower_time_window, upper_time_window]
lower_voltage_window = 200  # [V]
duration_window = 200e-9  # [s]
# Find the time when voltage first reaches lower_voltage_window.
idx_min_wanted_time = np.where(voltages_raw_anode > lower_voltage_window)[0][0]
lower_time_window = times_raw_anode[idx_min_wanted_time]  # [s]
print(lower_time_window * 1e9)
upper_time_window = lower_time_window + duration_window  # [s]

# Limit the time to upper_time_window.
idx_min_wanted_time = np.where(times_raw_anode > lower_time_window)[0][0]
idx_max_wanted_time = np.where(times_raw_anode > upper_time_window)[0][0]

# Limit the time, voltages and currents to the wanted period.
times_expe_anode = (
    times_raw_anode[idx_min_wanted_time:idx_max_wanted_time]
    - lower_time_window
)
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

# Get absolute maximum of voltages and currents.
max_abs_voltage = np.max(np.abs(voltages_expe_anode))
max_abs_current = np.max(np.abs(currents_expe_anode))

# Plot the voltage and current signals.
fig, ax_v = plt.subplots()
# fig.suptitle("Voltage and current signals from Minesi2022")
# .. Plot the voltage signal.
ax_v.plot(
    times_expe_anode * 1e9,
    voltages_expe_anode * 1e-3,
    label="Voltage",
    color="black",
)
ax_v.set_xlim(0, duration_window * 1e9)
ax_v.set_ylim(-4, 8)
ax_v.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax_v.set_ylabel(r"$\mathregular{V \, [kV]}$")
# .. Add a second y-axis for the current signal.
ax_i = ax_v.twinx()
ax_i.plot(
    times_expe_anode * 1e9, currents_expe_anode, label="Current", color="red"
)
ax_i.set_ylabel(r"$\mathregular{I \, [A]}$", color="r")
ax_i.set_ylim(-25, 50)
ax_i.set_yticks([-25, -12.5, 0, 12.5, 25, 37.5, 50])
ax_i.grid(visible=False)
ax_i.spines["right"].set_color("r")
ax_i.tick_params(axis="y", color="r", labelcolor="r")

plt.show()


# %%
# Compute voltage and current at a given position.
# ------------------------------------------------

# Time vector for the simulation.
nb_steps = 1000
times = np.linspace(0, duration_window, nb_steps)  # [s]
# Position of probes for measurement
x = 6.1  # [m]


# Compute the voltage and current at probes' position.
solution.solve(x, times)

voltages = solution.voltage  # [V]
currents = solution.current  # [A]
energies = solution.energy  # [J]
xs = solution.x  # [m]
times = solution.t  # [s]


# %%
# Plot the voltage and current at a given position.
# -------------------------------------------------


# Do we want to plot the current and energy?
plot_current = True
plot_energy = True


# Compute the min and max values of the voltage and current,
# to have the same zero level for voltage and current.
max_abs_voltage = np.max(np.abs(voltages)) * 1.1
max_abs_current = np.max(np.abs(currents)) * 1.1

fig, ax_v = plt.subplots()

# Plot voltage.
plot_line_v = ax_v.plot(
    (times - x / c) * 1e9,
    voltages * 1e-3,
    color="k",
    ls="--",
    label="Voltage (computed)",
)
plot_line_v_measured = ax_v.plot(
    times_expe_anode * 1e9,
    voltages_expe_anode * 1e-3,
    color="k",
    label="Voltage (experimental)",
)
# .. Plot options for voltage.
ax_v.set_xlabel(r"$\mathregular{t - \frac{x_{meas}}{c} \, [ns]}$")
ax_v.set_ylabel(r"$\mathregular{V \, [kV]}$")
ax_v.set_ylim(-6, 8)
ax_v.spines["left"].set_color("k")
ax_v.set_xlim(times[0] * 1e9, times[-1] * 1e9)

# Plot current.
if plot_current:
    ax_i = ax_v.twinx()
    plot_line_i = ax_i.plot(
        (times - x / c) * 1e9,
        currents,
        color="r",
        ls="--",
        label="Current (computed)",
    )
    plot_line_i_measured = ax_i.plot(
        times_expe_anode * 1e9,
        currents_expe_anode,
        color="r",
        label="Current (experimental)",
    )
    # .. Plot options for current.
    ax_i.set_ylabel(r"$\mathregular{I \, [A]}$", color="r")
    # ax_i.set_ylim(-max_abs_current, max_abs_current)
    ax_i.set_ylim(-37.5, 50)
    ax_i.set_yticks([-37.5, -25, -12.5, 0, 12.5, 25, 37.5, 50])
    ax_i.grid(visible=False)
    # Change color of the right y-axis to red.
    ax_i.spines["right"].set_color("r")
    # Also change the color of the ticks.
    ax_i.tick_params(axis="y", colors="r")
    # Move x-position of the y-label.
    ax_i.yaxis.set_label_coords(1.05, 0.5)

# Plot energy.
if plot_energy:
    ax_e = ax_v.twinx()
    plot_line_e = ax_e.plot(
        (times - x / c) * 1e9,
        energies * 1e3,
        color="b",
        ls="--",
        label="Energy",
    )
    plot_line_e_measured = ax_e.plot(
        times_expe_anode * 1e9,
        energies_expe_anode * 1e3,
        color="b",
        label="Energy (experimental)",
    )
    # .. Plot options for energy.
    ax_e.set_ylabel(r"$\mathregular{E \, [mJ]}$", color="b")
    # Move the y-axis of ax_e to the right, by 100 points
    ax_e.spines["right"].set_position(("outward", 100))
    ax_e.grid(visible=False)
    ax_e.set_ylim(0, 2.5)
    # Change color of the right y-axis to blue.
    ax_e.spines["right"].set_color("b")
    # Also change the color of the ticks.
    ax_e.tick_params(axis="y", colors="b")


# .. Display all legends in the same box.
lines = plot_line_v + plot_line_v_measured
if plot_current and plot_line_i:
    lines += plot_line_i + plot_line_i_measured
if plot_energy and plot_line_e:
    lines += plot_line_e + plot_line_e_measured

labels_anode: list[str] = []
for line in lines:
    label = line.get_label()
    if isinstance(label, str):
        labels_anode.append(label)
    else:
        labels_anode.append("Unknown")
# ax_v.legend(lines, labels_anode, loc="lower right")

ax_v.legend(
    handles=plot_line_v + plot_line_v_measured,
    labels=["Computed", "Measured"],
    loc="lower right",
)

plt.show()


# # %%
# # Compute at different times for all positions and animate.
# # ---------------------------------------------------------

# # Define the space and time vectors.
# N_x = 100  # number of points in space.
# xs = np.linspace(0, L, N_x, dtype=float)  # Space vector [m]

# t_max = 200e-9  # Maximum time [s]
# N_t = 100  # Number of points in time.
# times = np.linspace(0, t_max, N_t, dtype=float)  # Time vector [s]

# solution.solve(xs, times)

# # Animate the voltage and current along the transmission line.
# ani = solution.animation()

# # Save the animation as a .mp4 file.
# ani.save("reproduce_Minesi2022_experiments.mp4", writer="ffmpeg", fps=20)

# # %%
# # Save the animation as a .gif file.
# import matplotlib.animation as animation  # noqa: E402

# writer = animation.PillowWriter(fps=15, bitrate=1800)
# ani.save("reproduce_Minesi2022_experiments.gif", writer=writer)

# %%
