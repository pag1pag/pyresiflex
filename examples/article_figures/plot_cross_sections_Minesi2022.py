r"""
Plotting cross sections from LXCat databases.
=============================================

In this example, 
"""  # noqa: D205

# %%
import matplotlib.pyplot as plt
import numpy as np

from pyresiflex.misc.plot import set_mpl_style
from pyresiflex.misc.read_lxcat_cross_section import (
    extract_energy_cross_section,
)
from pyresiflex.misc.utils import get_path_to_data
from pyresiflex.misc.units import e, m_e, k_b


# %%
# Mole fractions in an equilibrium mixture of CH4-air (Φ = 0.8) at 2000 and 3000 K,
# calculated with the CEA NASA code.
# Data from [Minesi2022]_, Table 2.
x_eq = {
    "N2": 0.66,
    "H2O": 0.094,
    "CO2": 0.028,
    "O2": 0.043,
    "CO": 0.044,
    "H2": 0.02,
    "H": 0.022,
    "NO": 0.02,
    # "OH": 0.037, --> No data available in LXCat.
}


# Sum of all mole fractions should be 1
sum_x_eq = sum(x_eq.values())
print(f"Sum of mole fractions: {sum_x_eq:.2%}")
# Missing fraction to make sum 1
missing_fraction = 1.0 - sum_x_eq
print(f"Missing fraction: {missing_fraction:.2%}")

# %%
# Possible databases in LXCat for each species
possible_databases = {
    "IAA": "black",
    "CCC": "red",
    "Biagi": "blue",
    "Flinders": "green",
    "Hayashi": "orange",
    "ISTLisbon": "purple",
    "Itikiwa": "brown",
    "Morgan": "pink",
}

# %%
# Plot cross sections for each species
for species in x_eq:
    fig, ax = plt.subplots()
    for db, color in possible_databases.items():
        try:
            energies, cross_sections = extract_energy_cross_section(
                get_path_to_data(f"cross_sections/{species}/{db}.txt")
            )
            ax.loglog(energies, cross_sections, label=db, linewidth=2, color=color)
        except FileNotFoundError:
            print(f"File not found for {species} in database {db}, skipping.")
            continue

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Momentum Transfer Cross Section (m²)")
    ax.set_title(f"Cross Sections for {species}")
    ax.legend()
    plt.show()


# %%
# Choose database for each species
# By default, choose Itikiwa for all species since it has data for the most species.
chosen_databases = {species: "Itikiwa" for species in x_eq}
# Override for specific species if needed
chosen_databases["H"] = "Morgan"  # Only Morgan has data for H.

# %%
# Plot total cross section, between 1 eV and 10 eV.
# Interpolate each cross section to a common energy grid.
# Interpolation preserves the left and right values (no extrapolation).

set_mpl_style(nb_columns=2)
fig, ax = plt.subplots()

xmin = 0.1 # eV
xmax = 10 # eV
energy_grid = np.linspace(xmin, xmax, 1000)  # eV
total_cross_section = np.zeros_like(energy_grid)  # m²

for species, db in chosen_databases.items():
    energies, cross_sections = extract_energy_cross_section(
        get_path_to_data(f"cross_sections/{species}/{db}.txt")
    )
    # Interpolate to common energy grid.
    interp_cross_sections = np.interp(energy_grid, energies, cross_sections)
    # Weight by mole fraction.
    weighted_cross_sections = interp_cross_sections * x_eq[species]
    total_cross_section += weighted_cross_sections
    # Plot individual (interpolated) weighted cross sections.
    ax.plot(
        energy_grid,
        weighted_cross_sections,
        label=f"{species} ({db})",
        lw=4,
        alpha=0.4,
        ls="--",
    )
    # Also plot raw data points for each species.
    ax.plot(
        energies,
        np.array(cross_sections) * x_eq[species],
        lw=4,
        alpha=0.6,
        ls="-",
        color=ax.get_lines()[-1].get_color(),
    )
    # Annotate species name at their maximum value (between 2 and 8 eV).
    mask = (energy_grid >= 3) & (energy_grid <= 7)
    max_idx = np.argmax(weighted_cross_sections[mask]) + np.argmax(mask)
    if species == "H":
        # Select a different position for H to avoid overlap.
        # Choose the point at 9 eV.
        max_idx = np.argmin(np.abs(energy_grid - 9))
    ax.annotate(
        species,
        xy=(energy_grid[max_idx], weighted_cross_sections[max_idx]),
        xytext=(0, -10),
        textcoords="offset points",
        fontsize=30,
        color=ax.get_lines()[-1].get_color(),
        fontweight="bold",
        # Add a white box around the text for better visibility.
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
    )
ax.plot(
    energy_grid,
    total_cross_section,
    label="Total",
    color="black",
)
ax.annotate(
    "Total",
    xy=(5, 1e-19),
    xytext=(0, 0),
    textcoords="offset points",
    fontsize=30,
    color="black",
    fontweight="bold",
)

ax.set_yscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(8e-22, 2e-19)
ax.set_xlabel(r"$\mathregular{\varepsilon \, [eV]}$")
ax.set_ylabel(r"$\mathregular{Q \, [m^2]}$")
plt.show()

# %%

# Get the temperature from the energy grid.
# E = (3/2) k_b T  -->  T = (2/3) E / k_b
# with E in J, k_b in J/K, T in K.
# Convert energy grid from eV to J.
T_e = (2 / 3) * (energy_grid * e) / k_b  # K

# Get the thermal velocity from the mean energy.
v_th = np.sqrt((8 * k_b * T_e) / (np.pi * m_e))  # m/s

n_n = 2.5e24 # m^-3, neutral density at 1 atm and 3000 K.

# Get the momentum transfer frequency.
nu_m = n_n * total_cross_section * v_th  # s^-1

fig, ax = plt.subplots()
ax.plot(T_e, nu_m, color="black", lw=4)
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlim(10_000, 50_000)
# ax.set_ylim(1e6, 5e10)
ax.set_xlabel(r"$\mathregular{T_e \, [K]}$")
ax.set_ylabel(r"$\mathregular{\bar{\nu_{eN}} \, [Hz]}$")
plt.show()
# %%
