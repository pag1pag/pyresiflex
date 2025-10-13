import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyresiflex.misc.utils import get_path_to_data


def set_mpl_style(nb_columns: int = 2) -> None:
    """Set the matplotlib style for plots.

    Parameters
    ----------
    nb_columns : int, optional
        Number of columns for the figure style.
        Options are 1 or 2. By default 2.
    """
    if nb_columns == 1:
        plt.style.use(get_path_to_data("article_one_column_figure.mplstyle"))
    elif nb_columns == 2:
        plt.style.use(get_path_to_data("article_two_columns_figure.mplstyle"))
    else:
        raise ValueError("`nb_columns` must be 1 or 2.")


# Time units.
converted_units_time = {
    "s": 1,
    "ms": 1e3,
    "us": 1e6,
    "ns": 1e9,
    "ps": 1e12,
}
# Voltage units.
converted_units_voltage = {
    "V": 1,
    "mV": 1e3,
    "kV": 1e-3,
}
# Current units.
converted_units_current = {
    "A": 1,
    "mA": 1e3,
    "kA": 1e-3,
}


def plot_voltage_current(
    voltage_time: np.ndarray,
    voltage_value: np.ndarray,
    current_time: np.ndarray,
    current_value: np.ndarray,
    fig_axes: tuple[Figure, Axes, Axes] | None = None,
    show: bool = False,
    voltage_time_unit: str = "ns",
    current_time_unit: str = "ns",
    voltage_value_unit: str = "kV",
    current_value_unit: str = "A",
):
    """Plot voltage and current signals on the same time axis.

    Parameters
    ----------
    voltage_time : numpy.ndarray
        Time values for the voltage signal.
    voltage_value : numpy.ndarray
        Voltage signal values.
    current_time : numpy.ndarray
        Time values for the current signal.
    current_value : numpy.ndarray
        Current signal values.
    fig_axes : tuple of Figure, Axes, Axes or None, optional
        A tuple containing the figure and two axes (one for voltage and one
        for current). If None, a new figure and axes are created.
        By default None
    show : bool, optional
        Whether to display the plot immediately. By default False
    voltage_time_unit : str, optional
        Unit for the voltage time axis.
        Options are 's', 'ms', 'us', 'ns', 'ps'.
        By default "ns"
    current_time_unit : str, optional
        Unit for the current time axis.
        Options are 's', 'ms', 'us', 'ns', 'ps'.
        By default "ns"
    voltage_value_unit : str, optional
        Unit for the voltage values.
        Options are 'V', 'mV', 'kV'. By default "kV"
    current_value_unit : str, optional
        Unit for the current values.
        Options are 'A', 'mA', 'kA'. By default "A"

    Returns
    -------
    tuple of Figure, Axes, Axes
        The figure and the two axes (one for voltage and one for current).
    """
    # Get back the fig and axes if provided.
    if fig_axes is None:
        fig, ax_v = plt.subplots()
        ax_i = ax_v.twinx()
    else:
        fig, ax_v, ax_i = fig_axes

    # Convert the units.
    if voltage_time_unit not in converted_units_time:
        raise ValueError(
            f"Invalid `voltage_time_unit` '{voltage_time_unit}'. "
            f"Choose from {list(converted_units_time.keys())}."
        )
    if current_time_unit not in converted_units_time:
        raise ValueError(
            f"Invalid `current_time_unit` '{current_time_unit}'. "
            f"Choose from {list(converted_units_time.keys())}."
        )
    if voltage_value_unit not in converted_units_voltage:
        raise ValueError(
            f"Invalid `voltage_value_unit` '{voltage_value_unit}'. "
            f"Choose from {list(converted_units_voltage.keys())}."
        )
    if current_value_unit not in converted_units_current:
        raise ValueError(
            f"Invalid `current_value_unit` '{current_value_unit}'. "
            f"Choose from {list(converted_units_current.keys())}."
        )
    voltage_time = voltage_time * converted_units_time[voltage_time_unit]
    current_time = current_time * converted_units_time[current_time_unit]
    voltage_value = voltage_value * converted_units_voltage[voltage_value_unit]
    current_value = current_value * converted_units_current[current_value_unit]

    # Plot the voltage signal.
    ax_v.plot(voltage_time, voltage_value, label="Voltage", color="black")
    ax_v.set_xlabel(rf"$\mathregular{{t \, [{voltage_time_unit}]}}$")
    ax_v.set_ylabel(rf"$\mathregular{{V \, [{voltage_value_unit}]}}$")

    # Plot the current signal.
    ax_i.plot(current_time, current_value, label="Current", color="r")
    ax_i.set_ylabel(
        rf"$\mathregular{{I \, [{current_value_unit}]}}$", color="r"
    )
    ax_i.grid(visible=False)
    ax_i.spines["right"].set_color("r")
    ax_i.tick_params(axis="y", color="r", labelcolor="r")

    if show:
        plt.show()
    return fig, ax_v, ax_i
