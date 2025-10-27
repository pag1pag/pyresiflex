import subprocess
import os
from pathlib import Path
import numpy as np

from pyresiflex.misc.utils import get_path_to_data


BOLSIG_DIR = get_path_to_data() / "Minesi2022" / "cross_sections"


def run(instruction_file: Path, verbose: bool = False):
    """Run Bolsigminus with the given instruction file.

    Parameters
    ----------
    instruction_file : Path
        Path to the Bolsigminus instruction file.
    verbose : bool, optional
        If True, print the stdout and stderr of the subprocess, by default False.
    """
    # Choose executable based on OS.
    if os.name == "nt":  # Windows.
        exec_cmd = BOLSIG_DIR / "bolsigminus.exe"
    else:
        exec_cmd = BOLSIG_DIR / "bolsigminus"

    if verbose:
        print(f"Running `{exec_cmd} {instruction_file}` in {BOLSIG_DIR}")

    # Run the subprocess.
    result = subprocess.run(
        [exec_cmd, str(instruction_file)],
        cwd=BOLSIG_DIR,
        shell=True,
        capture_output=True,
        text=True,
        input="\n",
    )
    if verbose:
        print("=" * 20)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        print("=" * 20)

    return result


def extract_bolsig_data(
    input_file: Path,
    output_file: Path,
    data: list[str] = ["Electric field / N (Td)"],
):
    """Extract data from a Bolsigminus output file.

    Parameters
    ----------
    input_file : Path
        Path to the input file.
    output_file : Path
        Path to the output file.
    data : list[str], optional
        List of data columns to extract, by default ["Electric field / N (Td)"]

    Raises
    ------
    ValueError
        If the input file is not in the expected format.
    """
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

    with open(input_file, "r") as f:
        lines = f.readlines()

    data_dict: dict[str, list[float]] = {key: [] for key in data}

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
    input_file = BOLSIG_DIR / "input_bolsig_Minesi2022.dat"
    run(input_file, verbose=False)

    # List of parameters to extract.
    parameters = [
        "Electric field / N (Td)",
        "Mobility *N (1/m/V/s)",
    ]

    # Files to process, as defined in the input file.
    file_names = [
        "2000K_Minesi2022.dat",
        "3000K_Minesi2022.dat",
    ]

    for file_name in file_names:
        bolsig_result = BOLSIG_DIR / file_name
        bolsig_result_extracted = BOLSIG_DIR / f"{file_name.replace('.dat', '.csv')}"

        extract_bolsig_data(
            bolsig_result,
            bolsig_result_extracted,
            data=parameters,
        )

        # Move input and output files into `results` folder
        output_folder = BOLSIG_DIR / "results"
        output_folder.mkdir(exist_ok=True)
        # Move (and erase if existing) files
        if (output_folder / bolsig_result.name).exists():
            (output_folder / bolsig_result.name).unlink()
        if (output_folder / bolsig_result_extracted.name).exists():
            (output_folder / bolsig_result_extracted.name).unlink()
        bolsig_result.rename(output_folder / bolsig_result.name)
        bolsig_result_extracted.rename(output_folder / bolsig_result_extracted.name)
