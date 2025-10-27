# Data looks like:
#
#   E/N (Td)	Mobility *N (1/m/V/s)
#   1.00000	0.810963E+24
#   2.18787	0.813074E+24
#   5.75149	0.831631E+24
#   ...   ...
#
#
#   E/N (Td)	Diffusion coefficient *N (1/m/s)
#   1.00000	0.261097E+25
#   2.18787	0.263258E+25
#   ...   ...
#
# This script extracts the data, and saves it in a more convenient format.

import numpy as np
from pyresiflex.misc.utils import get_path_to_data


def extract_bolsig_data(
    input_file,
    output_file,
    data=["Electric field / N (Td)"],
):
    with open(input_file, "r") as f:
        lines = f.readlines()

    data_dict = {key: [] for key in data}

    current_data_key = None
    for line in lines:
        line = line.strip()
        if line.startswith(f"E/N (Td)"):
            parts = line.split("\t")
            if len(parts) == 2 and parts[1] in data_dict:
                current_data_key = parts[1]
            else:
                current_data_key = None
            continue

        if current_data_key:
            parts = line.split("\t")
            if len(parts) == 2:
                data_dict[current_data_key].append(float(parts[1]))
                continue
            if line == "":
                current_data_key = None
                continue
            else:
                raise ValueError(f"Unexpected line format: {line}")
        else:
            continue

    data_dict_arrays = {}
    for key in data_dict:
        data_dict_arrays[key] = np.array(data_dict[key])

    # Save to output file in .csv format
    np.savetxt(
        output_file,
        np.column_stack([data_dict_arrays[key] for key in data]),
        delimiter=",",
        header=",".join(data),
        comments="",
        fmt="%.5E",  # Should be 1000.00, 0.339694E-12, 15.9093, 0.517766E+24
    )


if __name__ == "__main__":
    parameters = [
        "Electric field / N (Td)",
        "Momentum frequency /N (m3/s)",
        "Mean energy (eV)",
        "Mobility *N (1/m/V/s)",
    ]

    folders = [
        "2000K_no_ee_no_ei",
        "3000K_no_ee_no_ei",
        "3000K_ee_ei__ngas_2.5e24__ionisation_1e-1",
        "3000K_ee_ei__ngas_2.5e24__ionisation_1e-2",
        "3000K_ee_ei__ngas_2.5e24__ionisation_1e-3",
        "3000K_ee_ei__ngas_2.5e24__ionisation_1e-4",
        "3000K_ee_ei__ngas_2.5e24__ionisation_1e-5",
    ]

    for folder in folders:
        input_file = get_path_to_data(
            f"cross_sections/Bolsig/output/{folder}/output.dat"
        )
        output_file = get_path_to_data(
            f"cross_sections/Bolsig/output/{folder}/extracted_output.csv",
            force_return=True,
        )
        extract_bolsig_data(
            input_file,
            output_file,
            data=parameters,
        )
