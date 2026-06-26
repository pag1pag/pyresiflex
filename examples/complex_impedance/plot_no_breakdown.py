"""
Before a plasma breakdown: resistance in parallel with capacitance.
===================================================================

Before a plasma breakdown, the impedance of the gas can be represented
as a very high resistance (no ionization) in parallel to a constant
capacitance (representing the electrode and the surrounding environment).

Let us observe what are the current and voltage in this case.
"""  # noqa: D205

# This sets the following figure as the thumbnail for the example gallery.
# sphinx_gallery_thumbnail_number = 5

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
from pyresiflex.generator.generator_complex_impedance import GaussianGenerator
from pyresiflex.load.base_load import ComplexImpedanceBaseLoad
from pyresiflex.misc.units import c_0
from pyresiflex.misc.utils import get_path_to_data
from pyresiflex.solver.steady_impedance_solution import SteadyImpedanceSolution

plt.style.use(get_path_to_data("article_two_columns_figure.mplstyle"))

# %%
# Create a trapezoidal generator.
# -------------------------------

generator = GaussianGenerator(
    height=4e3,  # [V]
    mean=5e-9,  # [s]
    FWHM=5e-9,  # [s]
    R_g=1.0,  # [Ohm]
    C_g=1000.0e-12,  # [F]
)


# %%
# Transmission line parameters.
# -----------------------------

L = 5  # Length of the transmission line [m]
c = 0.67 * c_0  # Wave velocity [m/s]
Z_c = 100  # Characteristic impedance [Ohm]

cable = PerfectCable(
    L=L,
    Z_c=Z_c,
    c=c,
)


# %%
# Create a load class for no breakdown plasma.
# --------------------------------------------


class NoBreakdownPlasma(ComplexImpedanceBaseLoad):
    def __init__(self, R_l, C_l):
        super().__init__(purely_resistive=False)
        self.R_l = R_l
        self.C_l = C_l

    def load_impedance(self, frequency):
        # Load impedance is a capacitor in parallel with a resistor.
        return self.R_l / (
            1 + 1j * 2 * np.pi * frequency * self.R_l * self.C_l
        )


no_breakdown_plasma = NoBreakdownPlasma(
    R_l=1e6,  # Plasma resistance [Ohm], typical high value when no breakdown.
    C_l=10e-12,  # Plasma capacitance [F]
)


# %%
# Solve the system.
# -----------------

# Create the solution object with the generator, load and cable.
solution = SteadyImpedanceSolution(
    generator=generator,
    load=no_breakdown_plasma,
    cable=cable,
    nb_points_ft=30_000,  # Number of points for Fourier transform.
    max_time_ft=30_000e-9,  # Maximum time for Fourier transform [s]
)

x_meas_voltage = 0.85 * L  # Position of the measurement in meters.
x_meas_current = 0.85 * L  # Position of the measurement in meters.
N = 1000  # Number of points for time vector.
times = np.linspace(0, 100, N, dtype=float) * 1e-9

# Solve for voltage, current, and energy.
solution.solve(x_meas_voltage, t=times)
voltage = solution.voltage
current = solution.current
energy = solution.energy

# %%
# Plot voltage, current, and energy at remote configuration.
# ----------------------------------------------------------

fig, ax_v = plt.subplots()

# Plot voltage.
plot_line_v = ax_v.plot(
    times * 1e9,
    voltage * 1e-3,
    color="k",
)
# .. Plot options for voltage.
ax_v.set_xlabel(r"$\mathregular{t [ns]}$")
ax_v.set_ylabel(r"$\mathregular{V \, [kV]}$")
ax_v.set_ylim(-8, 8)
ax_v.spines["left"].set_color("k")
ax_v.set_xlim(times[0] * 1e9, times[-1] * 1e9)

# Plot current.
ax_i = ax_v.twinx()
ax_i.plot(
    times * 1e9,
    current,
    ls="--",
    color="r",
)
# .. Plot options for current.
ax_i.set_ylabel(r"$\mathregular{I \, [A]}$", color="r")
# ax_i.set_ylim(-max_abs_current, max_abs_current)
ax_i.set_ylim(-45, 45)
ax_i.grid(visible=False)
# Change color of the right y-axis to red.
ax_i.spines["right"].set_color("r")
# Also change the color of the ticks.
ax_i.tick_params(axis="y", colors="r")
# Move x-position of the y-label.
ax_i.yaxis.set_label_coords(1.05, 0.5)
# Set y-ticks for current.
ax_i.set_yticks([-45, -30, -15, 0, 15, 30, 45])

# Plot energy.
ax_e = ax_v.twinx()
ax_e.plot(
    times * 1e9,
    energy * 1e3,
    color="b",
    ls="-.",
    label="Energy",
    lw=2,
)
# .. Plot options for energy.
ax_e.set_ylabel(r"$\mathregular{E \, [mJ]}$", color="b")
# Move the y-axis of ax_e to the right, by 100 points
ax_e.spines["right"].set_position(("outward", 100))
ax_e.grid(visible=False)
ax_e.set_ylim(0, 1.0)
ax_e.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
# Change color of the right y-axis to blue.
ax_e.spines["right"].set_color("b")
# Also change the color of the ticks.
ax_e.tick_params(axis="y", colors="b")

plt.show()

# %%
