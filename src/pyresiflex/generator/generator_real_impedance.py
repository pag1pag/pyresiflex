from typing import Callable

import numpy as np

from pyresiflex.generator.base_generator import PurelyResistiveBaseGenerator


class TrapezoidalGenerator(PurelyResistiveBaseGenerator):
    """Trapezoidal generator.

    The generator impedance is a constant resistance.
    The generator voltage is a trapezoidal pulse.

    Parameters
    ----------
    R_g : float, optional
        Generator resistance, by default 1.0 Ohm.
    U_off : float, optional
        Voltage when the pulser is turned off, by default 0.0 V.
    U_on : float, optional
        Voltage when the pulser is turned on, by default 10e3 V.
    t_rise : float, optional
        Time to go from U_off to U_on, by default 3e-9 s.
    t_on : float, optional
        Time to stay at U_on, by default 10e-9 s.
    t_fall : float, optional
        Time to go from U_on to U_off, by default 3e-9 s.
    """

    def __init__(
        self,
        R_g: float = 1.0,
        U_off: float = 0.0,
        U_on: float = 10e3,
        t_rise: float = 3e-9,
        t_on: float = 10e-9,
        t_fall: float = 3e-9,
    ):
        super().__init__(R_g=R_g)

        # Generator voltage.
        self._U_off = U_off
        self._U_on = U_on
        self._t_rise = t_rise
        self._t_on = t_on
        self._t_fall = t_fall

    def generator_voltage(self, t: float) -> float:
        """Trapezoidal voltage pulse.

        First, the voltage is at U_off.
        Then, it rises to U_on in t_rise seconds.
        Next, it stays at U_on for t_on seconds.
        Finally, it falls to U_off in t_fall seconds.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Generator voltage in Volts.
        """
        if t < 0:
            return self._U_off
        elif t < self._t_rise:
            # Rising ramp.
            return self._U_off + (self._U_on - self._U_off) * t / self._t_rise
        elif t < self._t_rise + self._t_on:
            # Plateau.
            return self._U_on
        elif t < self._t_rise + self._t_on + self._t_fall:
            # Falling ramp.
            return (
                self._U_on
                - (self._U_on - self._U_off)
                * (t - self._t_rise - self._t_on)
                / self._t_fall
            )
        else:
            return self._U_off


class GaussianGenerator(PurelyResistiveBaseGenerator):
    """Gaussian generator.

    The generator impedance is a constant resistance.
    The generator voltage is a Gaussian pulse.

    Parameters
    ----------
    height : float
        Height of the Gaussian pulse [V].
    mean : float
        Mean (center) of the Gaussian pulse [s].
    FWHM : float
        Full width at half maximum of the Gaussian pulse [s].
    R_g : float, optional
        Generator resistance [Ohm], by default 0.0 Ohm.
    """

    def __init__(
        self,
        height: float,
        mean: float,
        FWHM: float,
        R_g: float = 0.0,
    ):
        super().__init__(R_g=R_g)

        # Generator voltage.
        self._height = height
        self._mean = mean
        self._FWHM = FWHM

    def generator_voltage(self, t: float) -> float:
        """Return the generator voltage as a Gaussian pulse.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Generator voltage in Volts.
        """
        return self.gaussian(
            t,
            height=self._height,  # [V]
            mean=self._mean,  # [s]
            FWHM=self._FWHM,  # [s]
        )

    @staticmethod
    def gaussian(t: float, height: float, mean: float, FWHM: float) -> float:
        r"""Return a Gaussian pulse.

        Parameters
        ----------
        t : float
            Time in seconds.
        height : float
            Height of the Gaussian pulse [V].
        mean : float
            Mean of the Gaussian pulse [s].
        FWHM : float
            Full width at half maximum of the Gaussian pulse [s].

        Returns
        -------
        float
            Value of the Gaussian pulse at time t [V].
        """
        sigma = FWHM * np.sqrt(2) / (np.sqrt(2 * np.log(2)) * 2)
        return height * np.exp(-(((t - mean) / sigma) ** 2))


class FromMeasurementGenerator(PurelyResistiveBaseGenerator):
    """Generator defined from a measurement function.

    Parameters
    ----------
    R_g : float
        Resistance of the generator [Ohm].
    V_meas : Callable
        Function that returns the generator voltage at a given time.
    """

    def __init__(self, R_g: float, V_meas: Callable[[float], float]):
        super().__init__(R_g=R_g)

        # Generator voltage.
        self._V_meas = V_meas

    def generator_voltage(self, t: float) -> float:
        """Return the generator voltage from a measurement function.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Generator voltage in Volts.
        """
        return self._V_meas(t)


class ConstantGenerator(PurelyResistiveBaseGenerator):
    """Constant generator.

    Parameters
    ----------
    R_g : float
        Resistance of the generator [Ohm].
    U_g : float
        Voltage of the generator [V].
    """

    def __init__(self, R_g: float, U_g: float):
        super().__init__(R_g=R_g)

        # Generator voltage.
        self.U_g = U_g

    def generator_voltage(self, t: float) -> float:
        """Return the generator voltage.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Generator voltage in Volts.
        """
        if t < 0:
            return 0.0
        return self.U_g
