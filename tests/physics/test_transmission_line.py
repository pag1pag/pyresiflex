import matplotlib.pyplot as plt
import numpy as np

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.generator_real_impedance import TrapezoidalGenerator
from pyresiflex.load.time_varying_resistance import ConstantResistance
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution


def test_transmission_line_no_reflection_long_cable(
    plot: bool = False,
    verbose: bool = False,
) -> None:
    """Test the transmission line when there is no reflection.

    The 'no-reflection' condition is achieved by using a very long cable.

    Parameters
    ----------
    plot : bool, optional
        If True, plot the results. Default is False.
    verbose : bool, optional
        If True, print the results. Default is False.
    """
    # .. Create a generator.
    generator = TrapezoidalGenerator(
        R_g=30.0,
        U_off=0.0,
        U_on=10e3,
        t_rise=3e-9,
        t_on=10e-9,
        t_fall=3e-9,
    )

    # .. Create a load.
    load = ConstantResistance(R=15.0)

    # .. Create a transmission line.
    cable = PerfectCable(
        L=100,  # Very long cable length for testing.
        Z_c=50.0,
        c=2.0e8,
    )

    # .. Create a solution for the transmission line.
    solution = PurelyResistiveSolution(
        cable=cable,
        generator=generator,
        load=load,
    )

    # .. Check the parameters.
    assert solution.generator == generator
    assert solution.load == load
    assert solution.cable == cable
    assert solution.L == 100.0
    assert solution.Z_c == 50.0
    assert solution.c == 2.0e8

    # .. Compute the voltage and current at a given position.
    x = 2  # Distance from the generator in meters.

    max_time = 100e-9  # 100 ns
    times = np.linspace(0, max_time, 1000, dtype=float)

    v_computed = np.zeros_like(times)
    i_computed = np.zeros_like(times)
    v_expected = np.zeros_like(times)
    i_expected = np.zeros_like(times)

    for i, t in enumerate(times):
        v_computed[i] = solution.V(x, t)
        i_computed[i] = solution.I(x, t)
        v_expected[i] = solution.alpha_g * generator.generator_voltage(
            t - x / solution.c
        )
        i_expected[i] = (
            solution.alpha_g
            * generator.generator_voltage(t - x / solution.c)
            / solution.Z_c
        )

    # .. Plot the results.
    if plot:
        plt.plot(times, v_computed, label="Computed Voltage")
        plt.plot(times, v_expected, label="Expected Voltage", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage on Transmission Line (No Reflection)")
        plt.legend()
        plt.grid()
        plt.show()
        plt.plot(times, i_computed, label="Computed Current")
        plt.plot(times, i_expected, label="Expected Current", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
        plt.title("Current on Transmission Line (No Reflection)")
        plt.legend()
        plt.grid()
        plt.show()

    # Check the results.
    for i, t in enumerate(times):
        if verbose:
            # Print the results for each time step.
            print(f"t = {t:.4e} s")
            print(
                f"V (computed) = {v_computed[i]:.4e} V, "
                f"V (expected) = {v_expected[i]:.4e} V"
            )
            print(
                f"I (computed) = {i_computed[i]:.4e} A, "
                f"I (expected) = {i_expected[i]:.4e} A"
            )
            print("-" * 60)

        atol = 1e-3
        rtol = 1e-3
        if v_expected[i] == 0:
            atol = 1
            rtol = 1e-1
        assert np.isclose(v_computed[i], v_expected[i], rtol=rtol, atol=atol)
        assert np.isclose(i_computed[i], i_expected[i], rtol=rtol, atol=atol)


def test_transmission_line_no_reflection_matched_impedance(
    plot: bool = False,
    verbose: bool = False,
) -> None:
    """Test the transmission line when there is no reflection.

    The 'no-reflection' condition is achieved by using matched impedance.

    Parameters
    ----------
    plot : bool, optional
        If True, plot the results. Default is False.
    verbose : bool, optional
        If True, print the results. Default is False.
    """
    # .. Create a generator.
    generator = TrapezoidalGenerator(
        R_g=50.0,
        U_off=0.0,
        U_on=10e3,
        t_rise=3e-9,
        t_on=10e-9,
        t_fall=3e-9,
    )

    # .. Create a load.
    load = ConstantResistance(R=50.0)

    # .. Create a transmission line.
    cable = PerfectCable(
        L=1,  # Short cable length for testing.
        Z_c=50.0,
        c=2.0e8,
    )

    # .. Create a solution for the transmission line.
    solution = PurelyResistiveSolution(
        cable=cable,
        generator=generator,
        load=load,
    )

    # .. Check the parameters.
    assert solution.generator == generator
    assert solution.load == load
    assert solution.cable == cable
    assert solution.L == 1.0
    assert solution.Z_c == 50.0
    assert solution.c == 2.0e8

    # .. Compute the voltage and current at a given position.
    x = 0.5  # Distance from the generator in meters.

    max_time = 20e-9  # 20 ns
    times = np.linspace(0, max_time, 1000, dtype=float)

    v_computed = np.zeros_like(times)
    i_computed = np.zeros_like(times)
    v_expected = np.zeros_like(times)
    i_expected = np.zeros_like(times)

    for i, t in enumerate(times):
        v_computed[i] = solution.V(x, t)
        i_computed[i] = solution.I(x, t)
        v_expected[i] = solution.alpha_g * generator.generator_voltage(
            t - x / solution.c
        )
        i_expected[i] = (
            solution.alpha_g
            * generator.generator_voltage(t - x / solution.c)
            / solution.Z_c
        )

    # .. Plot the results.
    if plot:
        plt.plot(times, v_computed, label="Computed Voltage")
        plt.plot(times, v_expected, label="Expected Voltage", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage on Transmission Line (No Reflection)")
        plt.legend()
        plt.grid()
        plt.show()
        plt.plot(times, i_computed, label="Computed Current")
        plt.plot(times, i_expected, label="Expected Current", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
        plt.title("Current on Transmission Line (No Reflection)")
        plt.legend()
        plt.grid()
        plt.show()

    # .. Check the results.
    # Check the maximum values.
    assert np.isclose(
        np.max(v_computed), np.max(v_expected), rtol=1e-2, atol=1e-2
    )
    assert np.isclose(
        np.max(i_computed), np.max(i_expected), rtol=1e-2, atol=1e-2
    )
    # Check the L2 norm of the difference.
    if verbose:
        print("L2 norm of the difference:")
        print(
            f"Voltage: {np.linalg.norm(v_computed - v_expected):.4e}, "
            f"Current: {np.linalg.norm(i_computed - i_expected):.4e}"
        )
    assert np.isclose(np.linalg.norm(v_computed - v_expected), 0.0, atol=1e1)
    assert np.isclose(np.linalg.norm(i_computed - i_expected), 0.0, atol=2e-1)


if __name__ == "__main__":
    print("Running tests...")
    # Uncomment the test you want to run.

    print("Testing long cable...")
    test_transmission_line_no_reflection_long_cable(plot=True, verbose=True)
    print("Testing matched impedance...")
    test_transmission_line_no_reflection_matched_impedance(
        plot=True, verbose=True
    )
