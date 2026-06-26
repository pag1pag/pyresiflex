"""
Constant resistance load simulation example.
============================================

This script demonstrates the simulation of a transmission line with a constant
resistance load using the `pyresiflex` library.

It models a simple circuit consisting of a generator,
a transmission line (cable), and two different constant resistance loads.

The voltage and current at the midpoint of the cable are computed and plotted
for both load cases.

The main steps in the script are:

- Importing required libraries.
- Defining transmission line parameters.
- Setting up the generator.
- Defining constant resistance loads.
- Creating solution objects for each load.
- Computing voltage and current at a specific position over time.
- Plotting the results.
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
from pyresiflex.generator.generator_real_impedance import ConstantGenerator
from pyresiflex.load.time_varying_resistance import ConstantResistance
from pyresiflex.misc.plot import set_mpl_style
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

set_mpl_style(nb_columns=1)

# %%
# Transmission line parameters
# ----------------------------

# Length of the transmission line.
L = 6  # [m]
# Maximum time for the simulation.
max_time = 1000  # [ns]

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
U_g = 5e3  # On voltage [V]

trapezoidal_generator = ConstantGenerator(R_g=R_g, U_g=U_g)


# %%
# Load parameters
# ---------------

Z_low = 10  # Low impedance [Ohm]
Z_high = 1e2  # High impedance [Ohm]

plasma_high_resistance = ConstantResistance(R=Z_high)
plasma_low_resistance = ConstantResistance(R=Z_low)

plasma_loads: list[ConstantResistance] = [
    plasma_high_resistance,
    plasma_low_resistance,
]
linestyles = [
    "-",
    ":",
]
labels = [
    f"Constant resistance at {Z_high} Ohm",
    f"Constant resistance at {Z_low} Ohm",
]

# %%
# Solution object.
# ----------------

solutions: list[PurelyResistiveSolution] = []
for plasma_load in plasma_loads:
    solutions.append(
        PurelyResistiveSolution(
            generator=trapezoidal_generator,
            load=plasma_load,
            cable=cable,
        )
    )

# %%
# Compute voltage and current at a given position.
# ------------------------------------------------

x_voltage = L / 2  # Position where voltage is computed [m]
x_current = L / 2  # Position where current is computed [m]
times = np.arange(0, max_time, 0.1) * 1e-9

# Compute the voltage and current at a given position.
voltages = []
currents = []
for solution in solutions:
    voltages.append(np.array([solution.V(x_voltage, t) for t in times]))
    currents.append(np.array([solution.I(x_current, t) for t in times]))

# %%
# Plot the voltage and current at a given position.
# -------------------------------------------------

fig, ax = plt.subplots()
ax_i = ax.twinx()


for voltage, current, linestyle, label in zip(
    voltages, currents, linestyles, labels
):
    # .. Plot voltage and current.
    ax.plot(
        times * 1e9,
        voltage * 1e-3,
        color="k",
        ls=linestyle,
        label=label,
    )
    ax_i.plot(
        times * 1e9,
        current,
        color="r",
        ls=linestyle,
    )


y_v_min, y_v_max = -0.5, 6.0
y_i_min, y_i_max = -50, 600

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

ax.legend(labels, fontsize=24, loc="right")

plt.show()

# # %%
# # Compute at different times for all positions and animate.
# # ---------------------------------------------------------

# # Define the space and time vectors.
# N_x = 100  # number of points in space.
# xs = np.linspace(0, L, N_x, dtype=float)  # Space vector [m]


# for solution, label, t_max in zip(solutions, labels, [200e-9, 1000e-9]):
#     print(f"Computing the solution for {label}...")

#     N_t = 100  # Number of points in time.
#     times = np.linspace(0, t_max, N_t, dtype=float)  # Time vector [s]

#     solution.solve(xs, times)

#     # Animate the voltage and current along the transmission line.
#     ani = solution.animation(
#         y_min_max_voltage=(-0.5, 5.5e3),
#         y_min_max_current=(-50, 550),
#     )

#     # Save the animation as a .mp4 file.
#     file_name = "animation_"
#     file_name += "_".join(label.lower().split(" "))

#     ani.save(
#         f"{file_name}.mp4",
#         writer="ffmpeg",
#         fps=15,
#     )

# %%
