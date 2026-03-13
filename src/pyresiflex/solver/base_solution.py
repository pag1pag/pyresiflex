from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.base_generator import BaseGenerator
from pyresiflex.load.base_load import BaseLoad


class BaseSolution(ABC):
    r"""Voltage and current solution within a cable.

    The cable is assumed to be a perfect transmission line, meaning that
    there is no loss in the line and that the wave velocity is constant.
    It is connected to a generator and a load.

    This abstract class defines the base solution.

    Parameters
    ----------
    cable : PerfectCable
        Perfect transmission line object.
    generator : BaseGenerator
        Generator object that provides the voltage source.
    load : BaseLoad
        Load object that provides the impedance at the end of the line.

    Notes
    -----
    The transmission line is assumed to be perfect, meaning that there is no
    loss in the line and that the wave velocity is constant.

    For a perfect transmission line, the voltage and current satisfy the wave
    equation (or telegrapher's equation without losses):

    .. math::

        \begin{align}
            \frac{\partial^2 V}{\partial t^2}(x, t)
            & = c^2 \frac{\partial^2 V}{\partial x^2}(x, t) \\
            \frac{\partial^2 I}{\partial t^2}(x, t)
            & = c^2 \frac{\partial^2 I}{\partial x^2}(x, t)
        \end{align}


    where:

    - :math:`V` is the voltage in Volt,
    - :math:`I` is the current in Ampere,
    - :math:`c` is the wave velocity in m/s,
    - :math:`x` is the position along the transmission line in meter,
    - :math:`t` is the time in second.

    To fully describe the behavior of the transmission line, we need to
    specify the boundary conditions at the ends of the transmission line.

    This is done is the derived classes, which implement the incident and
    reflected waves at the generator and the load.
    """

    def __init__(
        self,
        cable: PerfectCable,
        generator: BaseGenerator,
        load: BaseLoad,
    ):
        if not isinstance(cable, PerfectCable):
            raise TypeError("`cable` must be an instance of `PerfectCable`.")
        if not isinstance(generator, BaseGenerator):
            raise TypeError(
                "`generator` must be an instance of `BaseGenerator`."
            )
        if not isinstance(load, BaseLoad):
            raise TypeError("`load` must be an instance of `BaseLoad`.")

        self.cable: PerfectCable = cable
        """Perfect transmission line object."""
        self.generator: BaseGenerator = generator
        """Generator object that provides the voltage source."""
        self.load: BaseLoad = load
        """Load object that provides the impedance at the end of the line."""

        # Cable parameters.
        self.L: float = cable.L
        """Length of the transmission line in meters."""
        self.Z_c: float = cable.Z_c
        """Characteristic impedance of the transmission line in Ohm."""
        self.c: float = cable.c
        """Wave velocity in the transmission line in m/s."""

        # Store voltage, current, power and cumulative energy.
        self.voltage: np.ndarray
        self.current: np.ndarray
        self.power: np.ndarray
        self.energy: np.ndarray
        # Also store the time and position arrays for plotting.
        self.x: np.ndarray
        self.t: np.ndarray

    def N(self, t: float) -> int:
        r"""Get the number of wave generations that have occurred at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        int
            Number of wave generations.

        Notes
        -----
        The number of wave generations is given by:

        .. math::

            N(t) = \left\lfloor \frac{t}{2 L / c} \right\rfloor
        """
        L = self.cable.L
        c = self.cable.c
        N = np.floor(t / (2 * L / c)).astype(int)
        return N

    @abstractmethod
    def V_incident(self, t: float, n: int) -> float:
        r"""Incident wave.

        Compute the incident wave of generation :math:`n` at time :math:`t`.

        Parameters
        ----------
        t : float
            Time in seconds.
        n : int
            Generation number.

        Return
        ------
        float
            Incident wave value in volts.
        """
        raise NotImplementedError(
            "The incident wave is not implemented in GeneralSolution. "
            "Please use a different solution or implement the incident wave."
        )

    @abstractmethod
    def V_reflected(self, t: float, n: int) -> float:
        r"""Reflected wave.

        Compute the reflected wave of generation :math:`n` at time :math:`t`.

        Parameters
        ----------
        t : float
            Time in seconds. Can be equal to `t`, `t+x/c`, ...
        n : int
            Generation number.

        Return
        ------
        float
            Reflected wave value in volts.
        """
        raise NotImplementedError(
            "The reflected wave is not implemented in GeneralSolution. "
            "Please use a different solution or implement the reflected wave."
        )

    def V_incident_total(self, t: float) -> float:
        r"""Total incident voltage at time :math:`t`.

        Parameters
        ----------
        t : float
            Time in seconds.

        Return
        ------
        float
            Total incident voltage value in volts.

        Notes
        -----
        The total incident voltage is defined as:

        .. math::

            V_i(t) = \sum_{n=0}^{N(t)} V^n_i(t)

        where:

        - :math:`V^n_i(t)` is the incident voltage of generation :math:`n`.
        - :math:`N(t)` is the number of generations that have occurred at time
          :math:`t`.

        The number of generations is defined as:

        .. math::

            N(t) = \left\lfloor \frac{t}{\tau} \right\rfloor

        where:

        - :math:`\tau` is the time it takes for a wave to travel from the
          generator to the load and back, which is equal to :math:`2 L / c`.
        """
        # Get the number of generations.
        N_incident = self.N(t)

        # Compute the incident voltage for each generation.
        V_i = np.sum(
            np.array([self.V_incident(t, n) for n in range(N_incident + 1)])
        )
        return V_i

    def V_reflected_total(self, t: float) -> float:
        r"""Total reflected voltage at time :math:`t`.

        Parameters
        ----------
        t : float
            Time in seconds.

        Return
        ------
        float
            Total reflected voltage value in volts.

        Notes
        -----
        The total reflected voltage is defined as:

        .. math::

            V_r(t) = \sum_{n=0}^{N(t)} V^n_r(t)

        where:

        - :math:`V^n_r(t)` is the reflected voltage of generation :math:`n`.
        - :math:`N(t)` is the number of generations that have occurred at time
          :math:`t`.

        The number of generations is defined as:

        .. math::

            N(t) = \left\lfloor \frac{t}{\tau} \right\rfloor

        where:

        - :math:`\tau` is the time it takes for a wave to travel from the
          generator to the load and back, which is equal to :math:`2 L / c`.
        """
        # Get the number of generations.
        N_reflected = self.N(t)

        # Compute the reflected voltage for each generation.
        V_r = np.sum(
            np.array([self.V_reflected(t, n) for n in range(N_reflected + 1)])
        )
        return V_r

    def V(self, x: float, t: float) -> float:
        r"""Voltage at position :math:`x` and time :math:`t`.

        Parameters
        ----------
        x : float
            Position in meters.
        t : float
            Time in seconds.

        Return
        ------
        float
            Voltage value in volts.

        Notes
        -----
        The voltage is defined as:

        .. math::

            V(x, t) = V_i(t-x/c) + V_r(t+x/c)

        where:

        - :math:`V_i(t-x/c)` is the incident wave at time :math:`t-x/c`.
        - :math:`V_r(t+x/c)` is the reflected wave at time :math:`t+x/c`.
        - :math:`c` is the wave velocity in m/s.


        The total incident voltage is defined as:

        .. math::

            V_i(t-x/c) = \sum_{n=0}^{N(t-x/c)} V^n_i(t-x/c)

        where:

        - :math:`V^n_i(t-x/c)` is the incident voltage of generation :math:`n`.
        - :math:`N(t-x/c)` is the number of generations that have occurred at
          time :math:`t-x/c`.


        The total reflected voltage is defined as:

        .. math::

            V_r(t+x/c) = \sum_{n=0}^{N(t+x/c)} V^n_r(t+x/c)

        where:

        - :math:`V^n_r(t+x/c)` is the reflected voltage of
          generation :math:`n`.
        - :math:`N(t+x/c)` is the number of generations that
          have occurred at time :math:`t+x/c`.


        The number of generations is defined as:

        .. math::

            N(t) = \left\lfloor \frac{t}{\tau} \right\rfloor

        where:

        - :math:`\tau` is the time it takes for a wave to travel from the
          generator to the load and back, which is equal to :math:`2 L / c`.

        See Also
        --------
        :py:meth:`pyresiflex.solver.purely_resistive_solution.PurelyResistiveSolution.V_incident`
        :py:meth:`pyresiflex.solver.purely_resistive_solution.PurelyResistiveSolution.V_reflected`
        """
        # Compute the incident voltage for each generation.
        V_i = self.V_incident_total(t - x / self.c)
        # Compute the reflected voltage for each generation.
        V_r = self.V_reflected_total(t + x / self.c)

        # Return the total voltage.
        return V_i + V_r

    def I(self, x: float, t: float) -> float:  # noqa: E743
        r"""Intensity at position :math:`x` and time :math:`t`.

        Parameters
        ----------
        x : float
            Position in meters.
        t : float
            Time in seconds.

        Return
        ------
        float
            Intensity value in Ampere.

        Notes
        -----
        The current is defined as:

        .. math::

            I(x, t) = \frac{V_i(t-x/c) - V_r(t+x/c)}{Z_c}

        where:

        - :math:`V_i(t-x/c)` is the incident wave at time :math:`t-x/c`.
        - :math:`V_r(t+x/c)` is the reflected wave at time :math:`t+x/c`.
        - :math:`c` is the wave velocity in m/s.
        - :math:`Z_c` is the characteristic impedance of the transmission
          line in Ohm.


        The total incident voltage is defined as:

        .. math::

            V_i(t-x/c) = \sum_{n=0}^{N(t-x/c)} V^n_i(t-x/c)

        where:

        - :math:`V^n_i(t-x/c)` is the incident voltage of generation :math:`n`.
        - :math:`N(t-x/c)` is the number of generations that have occurred at
          time :math:`t-x/c`.


        The total reflected voltage is defined as:

        .. math::

            V_r(t+x/c) = \sum_{n=0}^{N(t+x/c)} V^n_r(t+x/c)

        where:

        - :math:`V^n_r(t+x/c)` is the reflected voltage of
          generation :math:`n`.
        - :math:`N(t+x/c)` is the number of generations that have occurred at
          time :math:`t+x/c`.


        The number of generations is defined as:

        .. math::

            N(t) = \left\lfloor \frac{t}{\tau} \right\rfloor

        where:

        - :math:`\tau` is the time it takes for a wave to travel from the
          generator to the load and back, which is equal to :math:`2 L / c`.

        See Also
        --------
        :py:meth:`pyresiflex.solver.purely_resistive_solution.PurelyResistiveSolution.V_incident`
        :py:meth:`pyresiflex.solver.purely_resistive_solution.PurelyResistiveSolution.V_reflected`
        """
        # Compute the incident voltage for each generation.
        V_i = self.V_incident_total(t - x / self.c)
        # Compute the reflected voltage for each generation.
        V_r = self.V_reflected_total(t + x / self.c)

        # Return the total current.
        return (V_i - V_r) / self.Z_c

    def solve(self, x: np.ndarray | float, t: np.ndarray | float) -> None:
        r"""Compute voltage, current and energy at position x and time t.

        Parameters
        ----------
        x : numpy.ndarray or float
            Position in meters. Can be a single value or an array of values.
            All values must be between 0 and L.
        t : numpy.ndarray or float
            Time in seconds. Can be a single value or an array of values.

        Notes
        -----
        The incident and reflected waves are computed for each generation
        and for each time. Then, the voltages and currents are computed
        from the incident and reflected waves.

        The voltages and currents are defined as:

        .. math::

            V(x, t) = \sum_{n=0}^{N(t-x/c)} V_i^n(t - x/c)
                    + \sum_{n=0}^{N(t+x/c)} V_r^n(t + x/c)

        .. math::

            I(x, t) = \sum_{n=0}^{N(t-x/c)} I_i^n(t - x/c)
                    + \sum_{n=0}^{N(t+x/c)} I_r^n(t + x/c)

        Power and cumulative energy are defined as:

        .. math::

            P(x, t) = V(x, t) I(x, t)

        .. math::

            E(x, t) = \int_0^t P(x, \tau) d\tau

        The cumulative energy is computed using the trapezoidal rule.

        See Also
        --------
        :py:meth:`pyresiflex.solver.purely_resistive_solution.PurelyResistiveSolution.V_incident`
        :py:meth:`pyresiflex.solver.purely_resistive_solution.PurelyResistiveSolution.V_reflected`
        :py:meth:`pyresiflex.solver.purely_resistive_solution.PurelyResistiveSolution.V`
        :py:meth:`pyresiflex.solver.purely_resistive_solution.PurelyResistiveSolution.I`
        :py:meth:`pyresiflex.solver.steady_impedance_solution.SteadyImpedanceSolution.V_incident`
        :py:meth:`pyresiflex.solver.steady_impedance_solution.SteadyImpedanceSolution.V_reflected`
        :py:meth:`pyresiflex.solver.steady_impedance_solution.SteadyImpedanceSolution.V`
        :py:meth:`pyresiflex.solver.steady_impedance_solution.SteadyImpedanceSolution.I`
        """
        # Ensure x is a 1D numpy array or a single value.
        x_array: np.ndarray
        if isinstance(x, (float, int)):
            x_array = np.array([x], dtype=float)
        elif isinstance(x, np.ndarray):
            if x.ndim != 1:
                raise ValueError("x must be a 1D array or a single value.")
            x_array = x.astype(float)
        else:
            raise ValueError("x must be a 1D array or a single value.")
        # Check that all x values are within the cable length.
        if np.any(x_array < 0) or np.any(x_array > self.L):
            raise ValueError("All x values must be between 0 and L.")

        # Ensure t is a 1D numpy array or a single value.
        t_array: np.ndarray
        if isinstance(t, (float, int)):
            t_array = np.array([t], dtype=float)
        elif isinstance(t, np.ndarray):
            if t.ndim != 1:
                raise ValueError("t must be a 1D array or a single value.")
            t_array = t.astype(float)
        else:
            raise ValueError("t must be a 1D array or a single value.")

        # Create 2D arrays for incident and reflected waves.
        V_incident = np.zeros((len(x_array), len(t_array)), dtype=float)
        V_reflected = np.zeros((len(x_array), len(t_array)), dtype=float)

        # Compute the incident and reflected waves for each position and time.
        for i, x_val in enumerate(x_array):
            for j, t_val in enumerate(t_array):
                # Compute the incident and reflected waves.
                V_incident[i, j] = self.V_incident_total(
                    t_val - x_val / self.c
                )
                V_reflected[i, j] = self.V_reflected_total(
                    t_val + x_val / self.c
                )

        # Compute the voltage and current for each position and time.
        voltage = V_incident + V_reflected
        current = (V_incident - V_reflected) / self.Z_c

        # Compute and store power and cumulative energy.
        power = voltage * current
        energy = np.zeros_like(voltage, dtype=float)
        for i in range(len(t_array)):
            energy[:, i] = np.trapezoid(
                power[:, : i + 1], t_array[: i + 1], axis=1
            )

        # Store the results.
        # If x is a single value, we need to reshape the arrays,
        # from 2D to 1D.
        if x_array.ndim == 1 and x_array.size == 1:
            voltage = voltage.flatten()
            current = current.flatten()
            power = power.flatten()
            energy = energy.flatten()
            x_array = x_array.flatten()

        self.voltage = voltage
        self.current = current
        self.power = power
        self.energy = energy

        # Store the position and time arrays for plotting.
        self.x = x_array
        self.t = t_array

    def animation(
        self,
        interval: int = 100,
        with_current: bool = True,
        y_min_max_voltage: tuple[float, float] | None = None,
        y_min_max_current: tuple[float, float] | None = None,
        show: bool = True,
    ) -> FuncAnimation:
        r"""Create an animation of the voltage and current over time.

        Parameters
        ----------
        interval : int, optional
            Delay between frames in milliseconds. Default is 100 ms.
        y_min_max_voltage : tuple of float or None, optional
            Tuple containing the minimum and maximum y-axis limits for the
            voltage plot in volts.
        y_min_max_current : tuple of float or None, optional
            Tuple containing the minimum and maximum y-axis limits for the
            current plot in amperes.
        show : bool, optional
            If True, display the animation using plt.show(). Default is True.
        """
        fig, ax_voltage = plt.subplots()

        # Set up the plot elements to be updated.
        # .. Title that will be updated.
        animation_title = ax_voltage.text(
            0.5,
            0.89,
            "",
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
            transform=ax_voltage.transAxes,
            ha="center",
        )
        # .. Voltage line that will be updated.
        (plot_line_voltage,) = ax_voltage.plot([], [], "k-", label="Voltage")
        # .. Current line that will be updated.
        if with_current:
            ax_current = ax_voltage.twinx()
            (plot_line_current,) = ax_current.plot(
                [], [], "r--", label="Current"
            )

        def _update_line(idx_t: int) -> tuple:
            # Update title with load impedance at current time.
            Z_l = self.load.load_impedance(self.t[idx_t])
            animation_title.set_text(
                f"t = {self.t[idx_t] * 1e9:.0f} ns\n"
                f"($Z_l$ = {Z_l:.1e} "
                r"$\Omega$)"
            )
            # Update voltage line.
            plot_line_voltage.set_data(self.x, self.voltage[:, idx_t] * 1e-3)

            if with_current:
                # Update current line.
                plot_line_current.set_data(self.x, self.current[:, idx_t])
                return plot_line_voltage, plot_line_current, animation_title
            else:
                return plot_line_voltage, animation_title

        # Set y-limits if provided for voltage.
        if y_min_max_voltage is None:
            max_abs_voltage = np.max(np.abs(self.voltage)) * 1.1 * 1e-3
            ax_voltage.set_ylim(-max_abs_voltage, max_abs_voltage)
        if isinstance(y_min_max_voltage, tuple):
            ax_voltage.set_ylim(y_min_max_voltage)
        else:
            raise ValueError(
                "`y_min_max_voltage` must be a tuple of 2 floats or None."
            )

        # Set y-limits if provided for current.
        if with_current:
            if y_min_max_current is None:
                max_abs_current = np.max(np.abs(self.current)) * 1.1
                ax_current.set_ylim(-max_abs_current, max_abs_current)
            if isinstance(y_min_max_current, tuple):
                ax_current.set_ylim(y_min_max_current)
            else:
                raise ValueError(
                    "`y_min_max_current` must be a tuple of 2 floats or None."
                )

        # Plot options..
        # .. for voltage.
        ax_voltage.set_xlim(0, self.L)
        ax_voltage.set_xlabel(r"$\mathregular{x \, [m]}$")
        ax_voltage.set_ylabel(r"$\mathregular{V \, [kV]}$")
        ax_voltage.grid(visible=True)
        if with_current:
            # Only add legends if both voltage and current are plotted.
            ax_voltage.legend(loc="upper left")
        # .. for current.
        if with_current:
            ax_current.set_ylabel(r"$\mathregular{I \, [A]}$", color="r")
            ax_current.grid(visible=False)
            ax_current.spines["right"].set_color("r")
            ax_current.tick_params(axis="y", colors="r", labelcolor="r")
            ax_current.legend(loc="upper right")

        # Create the animation.
        ani = FuncAnimation(
            fig,
            _update_line,
            frames=len(self.t),  # Number of frames.
            interval=interval,  # Interval between frames in ms.
            blit=True,  # Only redraw the parts that have changed.
        )

        if show:
            plt.show()

        return ani
