import numpy as np

from pyresiflex.generator.base_generator import ComplexImpedanceBaseGenerator


class TrapezoidalGenerator(ComplexImpedanceBaseGenerator):
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
    >>> from pyresiflex.generator.generator_complex_impedance import (
    ...     TrapezoidalGenerator,
    ... )
    >>> gen = TrapezoidalGenerator()
    >>> gen.generator_voltage(5e-9)
    10000.0
    >>> gen.generator_voltage(-1e-9)
    0.0

    .. minigallery::
        pyresiflex.generator.generator_complex_impedance.TrapezoidalGenerator
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
        super().__init__()

        self.R_g: float = R_g

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

    def generator_impedance(self, frequency: np.ndarray) -> np.ndarray:
        """Impedance of the generator.

        For a trapezoidal generator, the impedance is a constant resistance.

        Parameters
        ----------
        frequency : numpy.ndarray
            Frequency in Hz (not used for this generator).

        Returns
        -------
        numpy.ndarray
            Generator impedance in Ohm.

        Examples
        --------
        >>> import numpy as np
        >>> from pyresiflex.generator.generator_complex_impedance import (
        ...     TrapezoidalGenerator,
        ... )
        >>> gen = TrapezoidalGenerator(R_g=50.0)
        >>> freq = np.array([1e6, 2e6, 3e6])
        >>> bool(np.allclose(gen.generator_impedance(freq), 50.0))
        True
        """
        self.check_frequency(frequency)
        return self.R_g * np.ones_like(frequency)


class GaussianGenerator(ComplexImpedanceBaseGenerator):
    """Gaussian generator.

    The generator impedance can be a resistance, a capacitance, or both.
    The generator voltage is a Gaussian pulse.

    Parameters
    ----------
    height : float
        Height of the Gaussian pulse. [V]
    mean : float
        Mean (center) of the Gaussian pulse. [s]
    FWHM : float
        Full width at half maximum of the Gaussian pulse. [s]
    R_g : float, optional
        Generator resistance, by default 0.0 Ohm.
    C_g : float, optional
        Generator capacitance, by default 0.0 F.

    Examples
    --------
    .. minigallery::
        pyresiflex.generator.generator_complex_impedance.GaussianGenerator
    """

    def __init__(
        self,
        height: float,
        mean: float,
        FWHM: float,
        R_g: float = 0.0,
        C_g: float = 0.0,
    ):
        super().__init__()
        self.height = height
        self.mean = mean
        self.FWHM = FWHM
        self.R_g = R_g
        self.C_g = C_g
        if self.R_g == 0.0 and self.C_g != 0.0:
            raise ValueError(
                "If the generator has a capacitance, it must also have a "
                "resistance, to avoid numerical issues."
            )

    def generator_impedance(self, frequency: np.ndarray) -> np.ndarray:
        """Return the generator impedance.

        Parameters
        ----------
        frequency : numpy.ndarray
            In case the generator impedance depends on frequency.
            This can be useful if there is, for instance, a capacitor at the
            output of the generator.
            Here, it is not used.

        Returns
        -------
        numpy.ndarray
            The generator impedance.

        Examples
        --------
        >>> import numpy as np
        >>> from pyresiflex.generator.generator_complex_impedance import (
        ...     GaussianGenerator,
        ... )
        >>> gen = GaussianGenerator(height=1.0, mean=0.0, FWHM=1e-9, R_g=50.0)
        >>> freq = np.array([1e6, 2e6, 3e6])
        >>> bool(np.allclose(gen.generator_impedance(freq), 50.0))
        True
        """
        self.check_frequency(frequency)
        if self.R_g == 0.0 and self.C_g == 0.0:
            # Ideal voltage source.
            return np.zeros_like(frequency)
        elif self.C_g == 0.0:
            # Purely resistive generator.
            return self.R_g * np.ones_like(frequency)
        elif self.R_g != 0.0 and self.C_g != 0.0:
            # Resistive and capacitive generator.
            omega = 2 * np.pi * frequency
            return self.R_g / (1 + 1j * omega * self.C_g * self.R_g)
        else:
            raise ValueError(
                "If the generator has a capacitance, it must also have a "
                "resistance, to avoid numerical issues."
            )

    def generator_voltage(self, t: float) -> float:
        """Return the generator voltage.

        Parameters
        ----------
        t : float
            Time array, in seconds.

        Returns
        -------
        float
            The generator voltage, in volts.

        Examples
        --------
        >>> from pyresiflex.generator.generator_complex_impedance import (
        ...     GaussianGenerator,
        ... )
        >>> gen = GaussianGenerator(height=2.0, mean=0.0, FWHM=1e-9)
        >>> round(float(gen.generator_voltage(0.0)), 3)
        2.0
        """
        return self.gaussian(
            t,
            height=self.height,  # [V]
            mean=self.mean,  # [s]
            FWHM=self.FWHM,  # [s]
        )

    @staticmethod
    def gaussian(t: float, height: float, mean: float, FWHM: float) -> float:
        """Return a Gaussian function.

        Parameters
        ----------
        t : float
            Time array, in seconds.
        height : float
            Height of the Gaussian pulse.
        mean : float
            Mean (center) of the Gaussian pulse.
        FWHM : float
            Full width at half maximum of the Gaussian pulse.

        Returns
        -------
        float
            The Gaussian pulse.

        Examples
        --------
        >>> import numpy as np
        >>> from pyresiflex.generator.generator_complex_impedance import (
        ...     GaussianGenerator,
        ... )
        >>> peak = GaussianGenerator.gaussian(0.0, 3.0, 0.0, 1e-9)
        >>> bool(np.isclose(peak, 3.0))
        True
        """
        sigma = FWHM * np.sqrt(2) / (np.sqrt(2 * np.log(2)) * 2)
        return height * np.exp(-(((t - mean) / sigma) ** 2))
