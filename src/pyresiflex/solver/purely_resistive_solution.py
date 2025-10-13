from typing import Callable

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.base_generator import PurelyResistiveBaseGenerator
from pyresiflex.load.base_load import BaseResistance
from pyresiflex.solver.base_solution import BaseSolution


class PurelyResistiveSolution(BaseSolution):
    r"""Voltage and current solution for a cable with resistive boundaries.

    The cable is assumed to be a perfect transmission line, meaning that
    there is no loss in the line and that the wave velocity is constant.
    It is connected to a generator and a load. The generator provides
    a voltage source and a constant resistance, while the load provides
    a time-varying resistance at the end of the line.

    This class provides the analytical solution for the voltage and current
    in the transmission line, taking into account the generator and load
    impedances. The solution is based on the wave equation and the boundary
    conditions imposed by the generator and load.

    Parameters
    ----------
    cable : PerfectCable
        Perfect transmission line object.
    generator : PurelyResistiveBaseGenerator
        Generator object that provides the voltage source.
    load : BaseResistance
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


    At the generator end (x = 0), the boundary condition is given by:

    .. math::

        V(0, t) = V_g(t) - I(0, t) R_g

    where:

    - :math:`V_g(t)` is the generator voltage at time :math:`t`,
    - :math:`R_g` is the generator resistance in Ohm.


    At the load end (x = L), the boundary condition is given by:

    .. math::

        V(L, t) = I(L, t) R_l(t)

    where:

    - :math:`R_l(t)` is the load resistance at time :math:`t`.
    """

    def __init__(
        self,
        cable: PerfectCable,
        generator: PurelyResistiveBaseGenerator,
        load: BaseResistance,
    ):
        if not isinstance(cable, PerfectCable):
            raise TypeError("`cable` must be an instance of `PerfectCable`.")
        if not isinstance(generator, PurelyResistiveBaseGenerator):
            raise TypeError(
                "`generator` must be an instance of"
                " `PurelyResistiveBaseGenerator`."
            )
        if not isinstance(load, BaseResistance):
            raise TypeError("`load` must be an instance of `BaseResistance`.")

        if not load.purely_resistive:
            raise ValueError(
                "The load must be purely resistive for this solution."
            )
        if not generator.purely_resistive:
            raise ValueError(
                "The generator must be purely resistive for this solution."
            )

        super().__init__(cable, generator, load)

        # Generator parameters.
        self.R_g: float = generator.generator_impedance()
        self.V_g: Callable[[float], float] = generator.generator_voltage

        self.alpha_g: float = 1 / (1 + self.R_g / self.Z_c)
        """Generator resistance voltage divider."""

        # Load parameters.
        self.R_l: Callable[[float], float] = load.load_impedance

    def gamma_g(self) -> float:
        r"""Calculate the generator reflection coefficient.

        Return
        ------
        float
            Generator reflection coefficient, dimensionless.

        Notes
        -----
        The generator reflection coefficient is defined as:

        .. math::

            \Gamma_g = \frac{R_g - Z_c}{R_g + Z_c}

        where:

        - :math:`R_g` is the generator resistance in Ohm,
        - :math:`Z_c` is the characteristic impedance of the transmission line
          in Ohm.
        """
        return (self.R_g - self.Z_c) / (self.R_g + self.Z_c)

    def gamma_l(self, t: float) -> float:
        r"""Calculate the load reflection coefficient at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Return
        ------
        float
            Generator reflection coefficient, dimensionless.

        Notes
        -----
        The load reflection coefficient is defined as:

        .. math::

            \Gamma_L(t) = \frac{R_l(t) - Z_c}{R_l(t) + Z_c}

        where:

        - :math:`R_l(t)` is the load resistance at time :math:`t` in Ohm,
        - :math:`Z_c` is the characteristic impedance of the transmission line
          in Ohm.
        """
        return (self.R_l(t) - self.Z_c) / (self.R_l(t) + self.Z_c)

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

        Notes
        -----
        The incident wave of generation :math:`n` is defined as:

        .. math::

            V_i^n(t) = \alpha_g V_g\left(t-\frac{2 n L}{c}\right)
                \prod_{k=1}^n\left[
                    \Gamma_g \Gamma_l\left(t-\frac{(2 k - 1) L}{c}\right)
                \right]

        where:

        - :math:`\alpha_g` is the generator resistance voltage divider,
        - :math:`V_g` is the (time-varying) generator voltage,
        - :math:`\Gamma_g` is the generator reflection coefficient,
        - :math:`\Gamma_l` is the (time-varying) load reflection coefficient,
        - :math:`L` is the length of the transmission line,
        - :math:`c` is the wave velocity in the transmission line,
        - :math:`n` is the generation number.
        """
        # Get the reflection coefficient at the generator.
        gamma_g = self.gamma_g()

        # Compute the incident wave.
        v_incident = self.alpha_g * self.generator.generator_voltage(
            t - 2 * n * self.L / self.c
        )

        for k in range(1, n + 1):
            # Get the reflection coefficient at the load.
            gamma_l = self.gamma_l(t - (2 * k - 1) * self.L / self.c)
            v_incident *= gamma_l * gamma_g

        return v_incident

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

        Notes
        -----
        The reflected wave of generation :math:`n` is defined as:

        .. math::

            V_r^n(t) = \Gamma_l\left(t-\frac{L}{c}\right)
                \alpha_g V_g\left(t-\frac{2(n+1) L}{c}\right)
                \prod_{k=1}^n\left[
                    \Gamma_g \Gamma_l\left(t-\frac{(2 k + 1) L}{c}\right)
                \right]

        where:

        - :math:`\alpha_g` is the generator resistance voltage divider,
        - :math:`V_g` is the (time-varying) generator voltage,
        - :math:`\Gamma_g` is the generator reflection coefficient,
        - :math:`\Gamma_l` is the (time-varying) load reflection coefficient,
        - :math:`L` is the length of the transmission line,
        - :math:`c` is the wave velocity in the transmission line,
        - :math:`n` is the generation number.
        """
        # Get the reflection coefficient at the generator.
        gamma_g = self.gamma_g()
        # Get the reflection coefficient at the load.
        gamma_l = self.gamma_l(t - self.L / self.c)

        # Compute the reflected wave.
        v_reflected = (
            self.alpha_g
            * gamma_l
            * self.generator.generator_voltage(
                t - 2 * (n + 1) * self.L / self.c
            )
        )

        for k in range(1, n + 1):
            # Get the reflection coefficient at the load.
            gamma_l = self.gamma_l(t - (2 * k + 1) * self.L / self.c)
            v_reflected *= gamma_l * gamma_g

        return v_reflected
