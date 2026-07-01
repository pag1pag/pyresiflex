"""
Smallest example with a time-varying load resistance.
=====================================================

This example demonstrates the use of a time-varying load resistance in a
purely resistive transmission line system. The system consists of:

- A perfect transmission line,
- A constant voltage generator,
- A time-varying load resistance.

"""  # noqa: D205

# %%
# Import necessary libraries.
# ---------------------------
import matplotlib.pyplot as plt
import numpy as np

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.generator_real_impedance import ConstantGenerator
from pyresiflex.load.time_varying_resistance import PlasmaResistanceLinearFall
from pyresiflex.misc.utils import get_root
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

# %%
# Create a purely resistive solution with a time-varying load resistance.
# -----------------------------------------------------------------------
solution = PurelyResistiveSolution(
    cable=PerfectCable(L=5, Z_c=75, c=2e8),
    generator=ConstantGenerator(R_g=1, U_g=5e3),
    load=PlasmaResistanceLinearFall(
        Z_start=1e2, Z_end=10, t_start_fall=20e-9, t_end_fall=30e-9
    ),
)

# %%
# Solve.
# ------

# Solve the system at specific time points.
times = np.linspace(0, 40e-9, 1000)
# Here, the solution is computed at 6 meters.
solution.solve(x=5, t=times)

# %%
# Plot the voltage response over time.
# ------------------------------------

fig, ax = plt.subplots()
ax.plot(times * 1e9, solution.voltage * 1e-3, color="k")
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{V \, [kV]}$")
ax.set_title("Load voltage against time")
plt.show()


# Save the figure in the .docs folder.
fig.savefig(get_root() / ".." / ".." / "docs" / "img" / "smallest_example.png")

# %%
