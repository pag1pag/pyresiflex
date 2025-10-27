import matplotlib.pyplot as plt
import numpy as np

from pyresiflex.misc.units import e, k_b
from pyresiflex.misc.utils import get_path_to_data

output_folders = get_path_to_data("cross_sections", "Bolsig", "output")

fig_E, ax_E = plt.subplots()
fig_T, ax_T = plt.subplots()

reduced = True
if not reduced:
    n_tot = 2.5e24  # m^-3, neutral density at 1 atm and 3000 K.

for folder in output_folders.iterdir():
    if not folder.is_dir():
        continue
    for file in folder.iterdir():
        if not file.name == "extracted_output.csv":
            continue
        name = folder.name
        data = np.loadtxt(file, delimiter=",", skiprows=1)
        E_N = data[:, 0]  # Td
        mean_energy_eV = data[:, 2]  # eV
        mobility_N = data[:, 3]  # (1/m/V/s)
        mean_energy_K = mean_energy_eV * e / k_b
        T = (2 / 3) * mean_energy_K  # in K

        if not reduced:
            mobility_N = mobility_N / n_tot  # in m^2/V/s

        ax_E.plot(E_N, mobility_N, label=name)
        ax_T.plot(T, mobility_N, label=name)

ax_E.set_title("Reduced momentum transfer frequency from Bolsig+")
ax_E.set_xlabel(r"$\mathregular{E/N \, [Td]}$")
if reduced:
    ax_E.set_ylabel(r"$\mathregular{\mu * N \, [1/m/V/s]}$")
else:
    ax_E.set_ylabel(r"$\mathregular{\mu \, [m^2/V/s]}$")
ax_E.set_xscale("log")
ax_E.legend()

ax_T.set_title("Momentum transfer frequency from Bolsig+")
ax_T.set_xlabel(r"$\mathregular{T_e \, [K]}$")
if reduced:
    ax_T.set_ylabel(r"$\mathregular{\mu * N \, [1/m/V/s]}$")
else:
    ax_T.set_ylabel(r"$\mathregular{\mu \, [m^2/V/s]}$")
# ax_T.set_xscale("log")
ax_T.set_xlim(0, 50_000)
ax_T.legend()

plt.show()
