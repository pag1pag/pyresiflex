import re

from pyresiflex.misc.utils import get_path_to_data


def extract_energy_cross_section(filename):
    energies = []
    cross_sections = []
    in_table = False
    dash_line = re.compile(r"-{5,}")
    number_line = re.compile(
        r"^\s*([+-]?\d\.\d+e[+-]?\d+)\s+([+-]?\d\.\d+e[+-]?\d+)\s*$"
    )

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if dash_line.match(line):
                if not in_table:
                    in_table = True
                else:
                    break
                continue
            if in_table:
                m = number_line.match(line)
                if m:
                    energies.append(float(m.group(1)))
                    cross_sections.append(float(m.group(2)))
    return energies, cross_sections


# Example usage:
if __name__ == "__main__":
    energies, cross_sections = extract_energy_cross_section(
        get_path_to_data("cross_sections/NO/IAA.txt")
    )
    for e, cs in zip(energies, cross_sections):
        print(f"{e:.6e}\t{cs:.6e}")
