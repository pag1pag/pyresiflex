import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyresiflex.misc.plot import (
    plot_voltage_current,
    save_figure,
    set_mpl_style,
)

matplotlib.use("Agg")  # Use non-interactive backend for testing


def test_plot_voltage_current_basic():
    voltage_time = np.linspace(0, 10, 100)
    voltage_value = np.sin(voltage_time)
    current_time = np.linspace(0, 10, 100)
    current_value = np.cos(current_time)

    fig, ax_v, ax_i = plot_voltage_current(
        voltage_time, voltage_value, current_time, current_value
    )

    # Check that the returned objects are correct types
    assert isinstance(fig, Figure)
    assert isinstance(ax_v, Axes)
    assert isinstance(ax_i, Axes)

    # Check that voltage and current lines are present
    voltage_lines = ax_v.get_lines()
    current_lines = ax_i.get_lines()
    assert len(voltage_lines) == 1
    assert len(current_lines) == 1

    # Check axis labels
    assert ax_v.get_xlabel() == r"$\mathregular{t \, [ns]}$"
    assert ax_v.get_ylabel() == r"$\mathregular{V \, [kV]}$"
    assert ax_i.get_ylabel() == r"$\mathregular{I \, [A]}$"


def test_plot_voltage_current_units():
    voltage_time = np.array([1, 2, 3])
    voltage_value = np.array([1, 2, 3])
    current_time = np.array([1, 2, 3])
    current_value = np.array([1, 2, 3])

    fig, ax_v, ax_i = plot_voltage_current(
        voltage_time,
        voltage_value,
        current_time,
        current_value,
        voltage_time_unit="us",
        current_time_unit="ms",
        voltage_value_unit="mV",
        current_value_unit="kA",
    )

    # Check axis labels reflect units
    assert ax_v.get_xlabel() == r"$\mathregular{t \, [us]}$"
    assert ax_v.get_ylabel() == r"$\mathregular{V \, [mV]}$"
    assert ax_i.get_ylabel() == r"$\mathregular{I \, [kA]}$"


def test_plot_voltage_current_show(monkeypatch):
    voltage_time = np.linspace(0, 1, 10)
    voltage_value = np.ones(10)
    current_time = np.linspace(0, 1, 10)
    current_value = np.ones(10)

    called = {}

    def fake_show():
        called["show"] = True

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)
    plot_voltage_current(
        voltage_time, voltage_value, current_time, current_value, show=True
    )
    assert called.get("show", False)


def test_plot_voltage_current_fig_axes():
    import matplotlib.pyplot as plt

    fig, ax_v = plt.subplots()
    ax_i = ax_v.twinx()
    voltage_time = np.linspace(0, 1, 10)
    voltage_value = np.ones(10)
    current_time = np.linspace(0, 1, 10)
    current_value = np.ones(10)

    fig2, ax_v2, ax_i2 = plot_voltage_current(
        voltage_time,
        voltage_value,
        current_time,
        current_value,
        fig_axes=(fig, ax_v, ax_i),
    )
    assert fig2 is fig
    assert ax_v2 is ax_v
    assert ax_i2 is ax_i


def test_plot_voltage_current_invalid_units():
    voltage_time = np.array([0, 1, 2])
    voltage_value = np.array([1, 2, 3])
    current_time = np.array([0, 1, 2])
    current_value = np.array([1, 2, 3])

    # Invalid voltage_time_unit
    try:
        plot_voltage_current(
            voltage_time,
            voltage_value,
            current_time,
            current_value,
            voltage_time_unit="invalid",
        )
    except ValueError as e:
        assert "Invalid `voltage_time_unit`" in str(e)
    else:
        assert False, "Expected `ValueError` for invalid `voltage_time_unit`"

    # Invalid current_time_unit
    try:
        plot_voltage_current(
            voltage_time,
            voltage_value,
            current_time,
            current_value,
            current_time_unit="invalid",
        )
    except ValueError as e:
        assert "Invalid `current_time_unit`" in str(e)
    else:
        assert False, "Expected `ValueError` for invalid `current_time_unit`"

    # Invalid voltage_value_unit
    try:
        plot_voltage_current(
            voltage_time,
            voltage_value,
            current_time,
            current_value,
            voltage_value_unit="invalid",
        )
    except ValueError as e:
        assert "Invalid `voltage_value_unit`" in str(e)
    else:
        assert False, "Expected `ValueError` for invalid `voltage_value_unit`"

    # Invalid current_value_unit
    try:
        plot_voltage_current(
            voltage_time,
            voltage_value,
            current_time,
            current_value,
            current_value_unit="invalid",
        )
    except ValueError as e:
        assert "Invalid `current_value_unit`" in str(e)
    else:
        assert False, "Expected `ValueError` for invalid `current_value_unit`"


def test_plot_voltage_current_unit_conversion():
    voltage_time = np.array([1, 2])
    voltage_value = np.array([1, 2])
    current_time = np.array([1, 2])
    current_value = np.array([1, 2])

    fig, ax_v, ax_i = plot_voltage_current(
        voltage_time,
        voltage_value,
        current_time,
        current_value,
        voltage_time_unit="ms",
        current_time_unit="us",
        voltage_value_unit="mV",
        current_value_unit="mA",
    )

    # Check that the data is converted correctly
    voltage_line = ax_v.get_lines()[0]
    current_line = ax_i.get_lines()[0]
    np.testing.assert_array_equal(voltage_line.get_xdata(), voltage_time * 1e3)
    np.testing.assert_array_equal(current_line.get_xdata(), current_time * 1e6)
    np.testing.assert_array_equal(
        voltage_line.get_ydata(), voltage_value * 1e3
    )
    np.testing.assert_array_equal(
        current_line.get_ydata(), current_value * 1e3
    )

    # Check axis labels
    assert ax_v.get_xlabel() == r"$\mathregular{t \, [ms]}$"
    assert ax_v.get_ylabel() == r"$\mathregular{V \, [mV]}$"
    assert ax_i.get_ylabel() == r"$\mathregular{I \, [mA]}$"


def test_plot_voltage_current_axis_colors():
    voltage_time = np.array([0, 1])
    voltage_value = np.array([1, 2])
    current_time = np.array([0, 1])
    current_value = np.array([1, 2])

    fig, ax_v, ax_i = plot_voltage_current(
        voltage_time, voltage_value, current_time, current_value
    )
    # Check that the right axis color is set to red
    assert colors.same_color(ax_i.spines["right"].get_edgecolor(), "r")
    # Check that tick labels are colored red
    for label in ax_i.get_yticklabels():
        assert colors.same_color(label.get_color(), "r")


def test_set_mpl_style_one_and_two_columns():
    # Both supported column counts apply a known matplotlib style file.
    set_mpl_style(nb_columns=1)
    set_mpl_style(nb_columns=2)


def test_set_mpl_style_invalid():
    with pytest.raises(ValueError):
        set_mpl_style(nb_columns=3)


def test_save_figure(monkeypatch, tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    saved = {}

    def fake_savefig(path, *args, **kwargs):
        saved["path"] = path

    # Point the figures directory at a fresh (non-existent) temp location so
    # the directory-creation branch is exercised.
    monkeypatch.setattr(
        "pyresiflex.misc.plot.get_root", lambda: tmp_path / "a" / "b"
    )
    monkeypatch.setattr(fig, "savefig", fake_savefig)
    save_figure(fig, "test_figure", dpi=100)
    assert str(saved["path"]).endswith("test_figure.png")
    assert (tmp_path / "figures").exists()
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
