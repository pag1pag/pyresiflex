from abc import ABC, abstractmethod

import numpy as np


class BaseGenerator(ABC):
    """Abstract class for a generator.

    A generator is defined by its impedance and its voltage.

    Parameters
    ----------
    purely_resistive : bool
        Flag to indicate if the generator impedance is purely resistive.
        If True, the generator is a constant resistance.
    """

    def __init__(self, purely_resistive: bool):
        super().__init__()
        self.purely_resistive: bool = purely_resistive
        """Flag to indicate if the generator impedance is purely resistive."""

    @abstractmethod
    def generator_voltage(self, t: float) -> float:
        r"""Voltage of the generator.

        The generator voltage can depend on the time.
        This function should be implemented in the derived class.

        Parameters
        ----------
        t : float
            Time in seconds.

        Return
        ------
        float
            Generator voltage in Volts.
        """
        raise NotImplementedError(
            "Define the generator voltage calculation in the derived class."
        )


class PurelyResistiveBaseGenerator(BaseGenerator):
    """A generator with a purely resistive impedance.

    Parameters
    ----------
    R_g : float
        Resistance in Ohm.
    """

    def __init__(self, R_g: float):
        super().__init__(purely_resistive=True)
        self.R_g: float = R_g
        """Generator impedance, here a constant resistance."""

    def generator_impedance(self) -> float:
        r"""Impedance of the generator.

        This function should be implemented in the derived class.

        Return
        ------
        float
            Generator impedance in Ohm.

        Notes
        -----
        For a constant resistance, the impedance is the same for all
        frequencies, therefore:

        .. math::

            Z_g(f) = R_g

        where:

        - :math:`R_g` is the generator resistance in Ohm.
        """
        return self.R_g


class ComplexImpedanceBaseGenerator(BaseGenerator):
    """A generator with a non-purely resistive impedance."""

    def __init__(self):
        super().__init__(purely_resistive=True)

    @abstractmethod
    def generator_impedance(self, frequency: np.ndarray) -> np.ndarray:
        r"""Impedance of the generator.

        This function should be implemented in the derived class.

        Parameters
        ----------
        frequency : numpy.ndarray
            Array of frequency in Hz.

        Return
        ------
        numpy.ndarray
            Array of generator impedance in Ohm, can be complex if the
            generator is a capacitor or an inductor. The impedance is the
            same for all frequencies if the generator is a resistance.

        Notes
        -----
        For a constant resistance, the impedance is the same for all
        frequencies, therefore:

        .. math::

            Z_g(f) = R_g

        where:

        - :math:`R_g` is the generator resistance in Ohm.


        For a capacitor, the impedance is:

        .. math::

            Z_g(f) = \frac{1}{j 2 \pi f C_g}

        where:

        - :math:`j` is the imaginary unit,
        - :math:`f` is the frequency in Hz,
        - :math:`C_g` is the capacitance in Farad.


        For an inductor, the impedance is:

        .. math::

            Z_g(f) = j 2 \pi f L_g

        where:

        - :math:`L_g` is the inductance in Henry.
        """
        self.check_frequency(frequency)

        raise NotImplementedError(
            "Define the generator impedance calculation in the derived class."
        )

    @staticmethod
    def check_frequency(frequency: np.ndarray) -> None:
        """Check that the frequency array is valid.

        Parameters
        ----------
        frequency : numpy.ndarray
            Array of frequency in Hz.

        Raises
        ------
        TypeError
            If frequency is not a numpy array.
        ValueError
            If frequency is not one-dimensional.
            If frequency values are not real.
            If frequency values are negative.
            If frequency values are not finite.
        """
        # Check that frequency is a numpy array.
        if not isinstance(frequency, np.ndarray):
            raise TypeError("Frequency must be a numpy array.")
        # Check that frequency is one-dimensional.
        if frequency.ndim != 1:
            raise ValueError("Frequency must be a one-dimensional array.")
        # Check that frequency values are real.
        if not np.isrealobj(frequency):
            raise ValueError("Frequency values must be real.")
        # Check that frequency values are finite.
        if not np.all(np.isfinite(frequency)):
            raise ValueError("Frequency values must be finite.")
