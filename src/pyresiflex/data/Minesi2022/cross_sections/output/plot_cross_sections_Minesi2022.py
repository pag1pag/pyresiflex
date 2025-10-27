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
from pyresiflex.misc.units import e, k_b, m_e
from pyresiflex.misc.utils import get_path_to_data

# %%
# Mole fractions in an equilibrium mixture of CH4-air (Φ = 0.8)
# at 2000 and 3000 K, calculated with the CEA NASA code.
# Data from [Minesi2022]_, Table 2.
x_eq = {
    2000: {
        "N2": 0.718,
        "H2O": 0.153,
        "CO2": 0.077,
        "O2": 0.037,
        "CO": 0.001,
        "H2": 0.00,
        "H": 0.00,
        "NO": 0.003,
        # "OH": 0.002, --> No data available in LXCat.
    },
    3000: {
        "N2": 0.66,
        "H2O": 0.094,
        "CO2": 0.028,
        "O2": 0.043,
        "CO": 0.044,
        "H2": 0.02,
        "H": 0.022,
        "NO": 0.02,
        # "OH": 0.037, --> No data available in LXCat.
    },
}


# Sum of all mole fractions should be 1
for T in x_eq:
    print(f"Mole fractions at {T} K:")
    for species, x in x_eq[T].items():
        print(f"  {species}: {x:.2%}")
    sum_x_eq = sum(x_eq[T].values())
    print(f"Sum of mole fractions: {sum_x_eq:.2%}")
    # Missing fraction to make sum 1
    missing_fraction = 1.0 - sum_x_eq
    print(f"Missing fraction: {missing_fraction:.2%}")
    print("-" * 30)

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

latex_labels = {
    "N2": r"$\mathrm{N_2}$",
    "H2O": r"$\mathrm{H_2O}$",
    "CO2": r"$\mathrm{CO_2}$",
    "O2": r"$\mathrm{O_2}$",
    "CO": r"$\mathrm{CO}$",
    "H2": r"$\mathrm{H_2}$",
    "H": r"$\mathrm{H}$",
    "NO": r"$\mathrm{NO}$",
    "OH": r"$\mathrm{OH}$",
}


# %%
# Plot cross sections for each species
for T in x_eq:
    for species in x_eq[T]:
        fig, ax = plt.subplots()
        for db, color in possible_databases.items():
            try:
                energies, cross_sections = extract_energy_cross_section(
                    get_path_to_data(f"cross_sections/{species}/{db}.txt")
                )
                ax.loglog(
                    energies,
                    cross_sections,
                    label=db,
                    linewidth=2,
                    color=color,
                )
            except FileNotFoundError:
                print(
                    f"File not found for {species} in database {db}, skipping."
                )
                continue

        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Momentum Transfer Cross Section (m²)")
        ax.set_title(f"Cross Sections for {species}")
        ax.legend()
        plt.show()


# %%
# Choose database for each species
# By default, choose Itikiwa for all species
# since it has data for the most species.
chosen_databases: dict[int, dict[str, str]] = {T: {} for T in x_eq}
for T in x_eq:
    for species in x_eq[T]:
        chosen_databases[T][species] = "Itikiwa"
        if species == "H":
            # Only Morgan has data for H.
            chosen_databases[T]["H"] = "Morgan"

# %%
# Plot total cross section, between 1 eV and 10 eV.
# Interpolate each cross section to a common energy grid.
# Interpolation preserves the left and right values (no extrapolation).

set_mpl_style(nb_columns=1)
fig, ax = plt.subplots()
xmin = 0.1  # eV
xmax = 10  # eV
energy_grid = np.linspace(xmin, xmax, 1000)  # eV

linestyle = {
    2000: ":",
    3000: "-",
}
colors = {
    "N2": "blue",
    "H2O": "red",
    "CO2": "brown",
    "O2": "green",
    "CO": "orange",
    "H2": "purple",
    "H": "magenta",
    "NO": "pink",
    "OH": "gray",
}

total_cross_sections: list[np.ndarray] = []


for T in chosen_databases:
    total_cross_section = np.zeros_like(energy_grid)  # m²

    for species, db in chosen_databases[T].items():
        energies, cross_sections = extract_energy_cross_section(
            get_path_to_data(f"cross_sections/{species}/{db}.txt")
        )
        # Interpolate to common energy grid.
        interp_cross_sections = np.interp(
            energy_grid, energies, cross_sections
        )
        # Weight by mole fraction.
        weighted_cross_sections = interp_cross_sections * x_eq[T][species]
        total_cross_section += weighted_cross_sections
        # Plot individual (interpolated) weighted cross sections.
        ax.plot(
            energy_grid,
            weighted_cross_sections,
            lw=2,
            alpha=0.4,
            ls="--",
            color=colors[species],
        )
        # Also plot raw data points for each species.
        ax.plot(
            energies,
            np.array(cross_sections) * x_eq[T][species],
            lw=4,
            alpha=0.6,
            ls=linestyle[T],
            color=ax.get_lines()[-1].get_color(),
        )
        if T == 3000:
            # Annotate species name at their max value (between 3 and 6 eV).
            mask = (energy_grid >= 3) & (energy_grid <= 6)
            max_idx = np.argmax(weighted_cross_sections[mask]) + np.argmax(
                mask
            )
            if species == "H":
                # Select a different position for H to avoid overlap.
                # Choose the point at 9 eV.
                max_idx = np.argmin(np.abs(energy_grid - 9))
            ax.annotate(
                f"{latex_labels[species]}",
                xy=(energy_grid[max_idx], weighted_cross_sections[max_idx]),
                xytext=(0, -10),
                textcoords="offset points",
                fontsize=32,
                color=ax.get_lines()[-1].get_color(),
                # fontweight="bold",
                # Add a white box around the text for better visibility.
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7
                ),
            )
    ax.plot(
        energy_grid,
        total_cross_section,
        color="black",
        ls=linestyle[T],
    )
    ax.annotate(
        "Total",
        xy=(5, 9e-20),
        xytext=(0, 0),
        textcoords="offset points",
        fontsize=32,
        color="black",
        fontweight="bold",
        # Add a white box around the text for better visibility.
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
    )

    total_cross_sections.append(total_cross_section)

ax.set_yscale("log")
ax.set_xlim(xmin, xmax)
ax.set_ylim(8e-22, 2e-19)
ax.set_xlabel(r"$\mathregular{\varepsilon \, [eV]}$")
ax.set_ylabel(r"$\mathregular{Q \, [m^2]}$")

(line_2000,) = ax.plot([], [], color="black", ls=":", lw=4, label="2000 K")
(line_3000,) = ax.plot([], [], color="black", ls="-", lw=4, label="3000 K")
ax.legend(handles=[line_2000, line_3000], loc="center right")
plt.show()

# %%

E_N_bolsig, nu_N_m_bolsig, mean_eV_bolsig = np.loadtxt(
    get_path_to_data("cross_sections", "output.csv"),
    delimiter=",",
    unpack=True,
    skiprows=1,
)

nu_m_bolsig = nu_N_m_bolsig * 2.5e24

print(E_N_bolsig, nu_m_bolsig, mean_eV_bolsig)

# %%

# Get the temperature from the energy grid.
# E = (3/2) k_b T  -->  T = (2/3) E / k_b
# with E in J, k_b in J/K, T in K.
# Convert energy grid from eV to J.
T_e = (2 / 3) * (energy_grid * e) / k_b  # K

# Get the thermal velocity from the mean energy.
v_th = np.sqrt((8 * k_b * T_e) / (np.pi * m_e))  # m/s

n_n = 2.5e24  # m^-3, neutral density at 1 atm and 3000 K.

# Get the momentum transfer frequency.

fig, ax = plt.subplots()

for i, T in enumerate(chosen_databases):
    total_cross_section = total_cross_sections[i]
    nu_m = n_n * total_cross_section * v_th  # s^-1
    ax.plot(
        T_e,
        nu_m,
        color="black",
        lw=4,
        ls=linestyle[T],
        label=f"{T} K",
    )

ax.plot(
    mean_eV_bolsig * (2 / 3) * (e / k_b),
    nu_m_bolsig,
    "ro",
    label="From Bolsig+",
)

ax.set_xlim(10_000, 50_000)
ax.set_ylim(0, 1e12)
ax.set_xlabel(r"$\mathregular{T_e \, [K]}$")
ax.set_ylabel(r"$\mathregular{\bar{\nu_{eN}} \, [Hz]}$")
ax.legend()
plt.show()
# %%
