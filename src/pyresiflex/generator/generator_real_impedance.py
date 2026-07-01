from typing import Callable

import numpy as np

from pyresiflex.generator.base_generator import PurelyResistiveBaseGenerator
from pyresiflex.misc.utils import gaussian_sigma_from_fwhm


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

    Examples
    --------
    .. minigallery::
        pyresiflex.generator.generator_real_impedance.TrapezoidalGenerator
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

        Examples
        --------
        >>> from pyresiflex.generator.generator_real_impedance import (
        ...     TrapezoidalGenerator,
        ... )
        >>> gen = TrapezoidalGenerator()
        >>> gen.generator_voltage(5e-9)
        10000.0
        >>> gen.generator_voltage(-1e-9)
        0.0
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

    Examples
    --------
    .. minigallery::
        pyresiflex.generator.generator_real_impedance.GaussianGenerator
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

        Examples
        --------
        >>> from pyresiflex.generator.generator_real_impedance import (
        ...     GaussianGenerator,
        ... )
        >>> gen = GaussianGenerator(height=2.0, mean=0.0, FWHM=1e-9)
        >>> round(float(gen.generator_voltage(0.0)), 3)
        2.0
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

        Examples
        --------
        >>> import numpy as np
        >>> from pyresiflex.generator.generator_real_impedance import (
        ...     GaussianGenerator,
        ... )
        >>> peak = GaussianGenerator.gaussian(0.0, 3.0, 0.0, 1e-9)
        >>> bool(np.isclose(peak, 3.0))
        True
        """
        sigma = gaussian_sigma_from_fwhm(FWHM)
        return height * np.exp(-(((t - mean) / sigma) ** 2))


class FromMeasurementGenerator(PurelyResistiveBaseGenerator):
    """Generator defined from a measurement function.

    Parameters
    ----------
    R_g : float
        Resistance of the generator [Ohm].
    V_meas : Callable
        Function that returns the generator voltage at a given time.

    Examples
    --------
    .. minigallery::
        pyresiflex.generator.generator_real_impedance.FromMeasurementGenerator
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

        Examples
        --------
        >>> from pyresiflex.generator.generator_real_impedance import (
        ...     FromMeasurementGenerator,
        ... )
        >>> gen = FromMeasurementGenerator(R_g=50.0, V_meas=lambda t: 2.0 * t)
        >>> gen.generator_voltage(3.0)
        6.0
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

    Examples
    --------
    .. minigallery::
        pyresiflex.generator.generator_real_impedance.ConstantGenerator
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

        Examples
        --------
        >>> from pyresiflex.generator.generator_real_impedance import (
        ...     ConstantGenerator,
        ... )
        >>> gen = ConstantGenerator(R_g=50.0, U_g=5.0)
        >>> gen.generator_voltage(1e-9)
        5.0
        >>> gen.generator_voltage(-1e-9)
        0.0
        """
        if t < 0:
            return 0.0
        return self.U_g
