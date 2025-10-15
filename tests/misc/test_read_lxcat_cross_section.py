import os
import tempfile

import pytest

from pyresiflex.misc.read_lxcat_cross_section import (
    extract_energy_cross_section,
)
from pyresiflex.misc.utils import get_path_to_data


def make_lxcat_file(contents):
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    tmp.write(contents)
    tmp.close()
    return tmp.name


def test_extract_energy_cross_section_basic():
    # Simulate a typical LXCat file with a table
    file_content = """
Header info
-----
 1.0000e+00  2.0000e-20
 2.0000e+00  3.0000e-20
 3.0000e+00  4.0000e-20
-----
Footer info
"""
    fname = make_lxcat_file(file_content)
    energies, cross_sections = extract_energy_cross_section(fname)
    os.unlink(fname)
    assert energies == [1.0, 2.0, 3.0]
    assert cross_sections == [2e-20, 3e-20, 4e-20]


def test_extract_energy_cross_section_ignores_non_table_lines():
    file_content = """
Some header
-----
 1.2345e+01  9.8765e-21
 2.3456e+01  8.7654e-21
-----
Some footer
"""
    fname = make_lxcat_file(file_content)
    energies, cross_sections = extract_energy_cross_section(fname)
    os.unlink(fname)
    assert energies == [12.345, 23.456]
    assert cross_sections == [9.8765e-21, 8.7654e-21]


def test_extract_energy_cross_section_empty_table():
    file_content = """
Header
-----
-----
Footer
"""
    fname = make_lxcat_file(file_content)
    energies, cross_sections = extract_energy_cross_section(fname)
    os.unlink(fname)
    assert energies == []
    assert cross_sections == []


def test_extract_energy_cross_section_handles_spaces_and_tabs():
    file_content = """
Header
-----
   1.0000e+00\t2.0000e-20
   2.0000e+00    3.0000e-20
-----
Footer
"""
    fname = make_lxcat_file(file_content)
    energies, cross_sections = extract_energy_cross_section(fname)
    os.unlink(fname)
    assert energies == [1.0, 2.0]
    assert cross_sections == [2e-20, 3e-20]


def test_extract_energy_cross_section_no_table():
    file_content = "No dashes or table here"
    fname = make_lxcat_file(file_content)
    energies, cross_sections = extract_energy_cross_section(fname)
    os.unlink(fname)
    assert energies == []
    assert cross_sections == []


def test_extract_energy_cross_section_real_file():
    fname = get_path_to_data("cross_sections/NO/IAA.txt")
    energies, cross_sections = extract_energy_cross_section(fname)
    assert len(energies) > 0
    assert len(cross_sections) > 0
    assert len(energies) == len(cross_sections)
    # Check first few values to ensure correct parsing
    assert energies[0] == pytest.approx(1.000000e-3, rel=1e-5)
    assert cross_sections[0] == pytest.approx(3.654450e-20, rel=1e-5)
    # Check all values
    energies_eV = [
        1.000000e-3,
        1.000000e-2,
        1.500000e0,
        3.000000e0,
        5.000000e0,
        7.500000e0,
        1.000000e1,
        1.500000e1,
        2.000000e1,
        3.000000e1,
        3.500000e1,
        4.000000e1,
        4.500000e1,
        5.000000e1,
        5.500000e1,
        6.000000e1,
        6.500000e1,
        7.000000e1,
        7.500000e1,
        8.000000e1,
        8.500000e1,
        9.000000e1,
        9.500000e1,
        1.000000e2,
        1.500000e2,
        2.000000e2,
        3.000000e2,
        4.000000e2,
        5.000000e2,
        7.500000e2,
        1.000000e3,
        2.000000e3,
        3.000000e3,
        4.000000e3,
        5.000000e3,
        6.000000e3,
        7.000000e3,
        8.000000e3,
        9.000000e3,
        1.000000e4,
        1.500000e4,
        2.000000e4,
        2.500000e4,
        3.000000e4,
        4.000000e4,
        5.000000e4,
        6.000000e4,
        7.000000e4,
        8.000000e4,
        9.000000e4,
        1.000000e5,
        2.000000e5,
        5.000000e5,
        1.000000e6,
        2.000000e6,
        5.000000e6,
        1.000000e7,
        2.000000e7,
        5.000000e7,
        1.000000e8,
        2.000000e8,
        5.000000e8,
        1.000000e9,
    ]

    cross_sections_cm2 = [
        3.654450e-20,
        5.192840e-20,
        1.013110e-19,
        8.255000e-20,
        7.405740e-20,
        7.234290e-20,
        7.348260e-20,
        7.283450e-20,
        6.744660e-20,
        6.030710e-20,
        5.436440e-20,
        5.017400e-20,
        4.638290e-20,
        4.273180e-20,
        3.925860e-20,
        3.605550e-20,
        3.318300e-20,
        2.985990e-20,
        2.734730e-20,
        2.513200e-20,
        2.318090e-20,
        2.145900e-20,
        1.993070e-20,
        1.857010e-20,
        1.162900e-20,
        8.387700e-21,
        4.778810e-21,
        3.387220e-21,
        2.597720e-21,
        1.453770e-21,
        9.528730e-22,
        3.219540e-22,
        1.614880e-22,
        9.821770e-23,
        6.660560e-23,
        4.842370e-23,
        3.695330e-23,
        2.922820e-23,
        2.376390e-23,
        1.697520e-23,
        9.366130e-24,
        5.580070e-24,
        3.734150e-24,
        2.690380e-24,
        1.606000e-24,
        1.078290e-24,
        7.800060e-25,
        5.940610e-25,
        4.698340e-25,
        3.824430e-25,
        3.184560e-25,
        9.830200e-26,
        2.277210e-26,
        7.892250e-27,
        2.699050e-27,
        6.014200e-28,
        1.805590e-28,
        5.193650e-29,
        9.595350e-30,
        2.623470e-30,
        7.097660e-31,
        1.246610e-31,
        3.323430e-32,
    ]

    for e, e_ref in zip(energies, energies_eV):
        assert e == pytest.approx(e_ref, rel=1e-5)
    for cs, cs_ref in zip(cross_sections, cross_sections_cm2):
        assert cs == pytest.approx(cs_ref, rel=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
