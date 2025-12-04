"""
Cable length influence on electrical signals.
=============================================

In experiments involving short pulse durations,
like in Nanosecond Repetitive Pulsed (NRP) discharges,
the influence of cable length on electrical signals is significant.

For long cable, a clear separation between incident and reflected
signals can be observed.
As the cable becomes shorter, superposition of signals will appear.

This example setups a simulation to visualize these effects, with the
following assumption

- Perfect transmission line behavior,
- Plasma load linearly decreasing,
- Trapezoidal pulse shape.
"""  # noqa: D205

# This sets the following figure as the thumbnail for the example gallery.
# sphinx_gallery_thumbnail_number = 3

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
from pyresiflex.generator.generator_real_impedance import TrapezoidalGenerator
from pyresiflex.load.time_varying_resistance import PlasmaResistanceLinearFall
from pyresiflex.misc.plot import set_mpl_style
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

set_mpl_style(nb_columns=1)

# %%
# Transmission line parameters
# ----------------------------

# Length of the transmission line.
# Test with 30 cm or 6 m.
L = 0.3  # 0.3 or 6 [m]
# Maximum time for the simulation.
max_time = 50  # 50 or 200 [ns]


c = 2e8  # Wave velocity [m/s]
Z_c = 75  # Characteristic impedance [Ohm]

cable = PerfectCable(
    L=L,
    Z_c=Z_c,
    c=c,
)

# %%
# Generator parameters
# --------------------

R_g = 1  # Impedance of the generator [Ohm]
U_off = 0.0  # Off voltage [V]
U_on = 10e3  # On voltage [V]
t_rise = 3e-9  # Rise time [s]
t_on = 10e-9  # Time on [s]
t_fall = 3e-9  # Fall time [s]

trapezoidal_generator = TrapezoidalGenerator(
    R_g=R_g,
    U_off=U_off,
    U_on=U_on,
    t_rise=t_rise,
    t_on=t_on,
    t_fall=t_fall,
)


# %%
# Load parameters
# ---------------

Z_start = 1e3  # Start impedance [Ohm]
Z_end = 10  # End impedance [Ohm]
t_start_fall = L / c + 5e-9  # Start fall time [s]
t_end_fall = L / c + 10e-9  # End fall time [s]

plasma_load = PlasmaResistanceLinearFall(
    Z_start=Z_start,
    Z_end=Z_end,
    t_start_fall=t_start_fall,
    t_end_fall=t_end_fall,
)

# %%
# Solution object.
# ----------------

solution = PurelyResistiveSolution(
    generator=trapezoidal_generator,
    load=plasma_load,
    cable=cable,
)


# %%
# Plot the generator voltage
# --------------------------

# Define time
times = np.arange(0, 100, 0.1) * 1e-9

generator_voltage = np.array(
    [trapezoidal_generator.generator_voltage(t) for t in times]
)

fig, ax = plt.subplots()
ax.plot(
    times * 1e9,
    generator_voltage * 1e-3,
    label="Pulser voltage",
)

ax.set_xlim(0, 100)
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{V \, [kV]}$")
# ax.set_title("Pulser voltage")
ax.legend()
ax.grid(visible=True)
plt.show()

# %%
# Plot the plasma resistance vs time.
# -----------------------------------

fig, ax = plt.subplots()

# Compute the plasma resistance at different times.
R_p = np.array([solution.R_l(t) for t in times])

ax.plot(
    times * 1e9,
    R_p,
    color="k",
    label="Plasma resistance",
)
ax.hlines(
    Z_c,
    xmin=0,
    xmax=100,
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

ax.set_xlim(0, 50)
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{Z \, [\Omega]}$")
ax.set_title("Resistance vs time")
ax.legend()
plt.show()


# %%
# Compute voltage and current at a given position.
# ------------------------------------------------

x_voltage = L / 2  # Position where voltage is computed [m]
x_current = L / 2  # Position where current is computed [m]
times = np.arange(0, max_time, 0.1) * 1e-9

# Compute the voltage and current at a given position.
voltages = np.array([solution.V(x_voltage, t) for t in times])
currents = np.array([solution.I(x_current, t) for t in times])

# %%
# Plot the voltage and current at a given position.
# -------------------------------------------------

fig, ax = plt.subplots(1, figsize=(20, 10))
ax_i = ax.twinx()

(line_v,) = ax.plot(times * 1e9, voltages * 1e-3, color="k", label="Voltage")
(line_i,) = ax_i.plot(
    times * 1e9,
    currents,
    color="r",
    label="Current",
)

print(
    f"Cable length L = {L:.1f} m\n"
    f"Voltage at x = {x_voltage:.2f} m, "
    f"Current at x = {x_current:.2f} m"
)

# Compute the min and max values of the voltage and current,
# to have the same zero level for voltage and current.
max_abs_voltage = np.max(np.abs(voltages)) * 1.1
max_abs_current = np.max(np.abs(currents)) * 1.1
min_max = "manual"
if min_max == "manual":
    # For L=0.3 m
    y_v_min, y_v_max = -5, 15
    y_i_min, y_i_max = -100, 300
    # For L=6.0 m
    # y_v_min, y_v_max = -10, 10
    # y_i_min, y_i_max = -150, 150
else:
    y_v_min, y_v_max = -max_abs_voltage * 1e-3, max_abs_voltage * 1e-3
    y_i_min, y_i_max = -max_abs_current, max_abs_current


ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
# .. Plot option for voltage.
ax.set_ylabel(r"$\mathregular{V \, [kV]}$")
ax.set_ylim(y_v_min, y_v_max)
# .. Plot option for current.
ax_i.set_ylabel(r"$\mathregular{I \, [A]}$", color="r")
ax_i.set_ylim(y_i_min, y_i_max)
ax_i.grid(visible=False)
ax_i.spines["right"].set_color("r")
ax_i.tick_params(axis="y", color="r", labelcolor="r")

ax.legend(handles=[line_v, line_i])

plt.show()

# %%
