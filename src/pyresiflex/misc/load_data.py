from dataclasses import dataclass

import yaml

from pyresiflex.misc.utils import get_path_to_data


@dataclass
class MinesiData:
    c: float
    """Wave propagation celerity in the transmission line [m/s]"""
    Δc: float
    """Uncertainty on the wave propagation celerity [m/s]"""
    Z_c: float
    """Characteristic impedance of the transmission line [Ω]"""
    ΔZ_c: float
    """Uncertainty on the characteristic impedance [Ω]"""
    L: float
    """Length of the transmission line [m]"""
    ΔL: float
    """Uncertainty on the length of the transmission line [m]"""
    x_meas: float
    """Position of the measurement probes [m]"""
    Δx_meas: float
    """Uncertainty on the position of the measurement probes [m]"""
    R_g: float
    """Internal resistance of the generator [Ω]"""
    ΔR_g: float
    """Uncertainty on the internal resistance of the generator [Ω]"""
    alpha_g: float
    """Attenuation factor of the generator-cable system [-]"""
    Δalpha_g: float
    """Uncertainty on the attenuation factor of generator-cable system [-]"""


def load_minesi_data() -> MinesiData:
    file_path = get_path_to_data("Minesi2022", "cable_properties.yaml")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return MinesiData(
        c=data["celerity"]["value"],
        Δc=data["celerity"]["uncertainty"],
        Z_c=data["characteristic_impedance"]["value"],
        ΔZ_c=data["characteristic_impedance"]["uncertainty"],
        L=data["length"]["value"],
        ΔL=data["length"]["uncertainty"],
        x_meas=data["x_meas"]["value"],
        Δx_meas=data["x_meas"]["uncertainty"],
        R_g=data["internal_resistance_generator"]["value"],
        ΔR_g=data["internal_resistance_generator"]["uncertainty"],
        alpha_g=data["attenuation_factor_generator_cable"]["value"],
        Δalpha_g=data["attenuation_factor_generator_cable"]["uncertainty"],
    )
