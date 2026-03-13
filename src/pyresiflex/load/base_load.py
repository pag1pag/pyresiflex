from abc import ABC, abstractmethod

import numpy as np


class BaseLoad(ABC):
    """Abstract class for a load.

    A load is defined by its impedance.
    The load impedance can depend on the time and the frequency.

    Parameters
    ----------
    purely_resistive : bool
        Flag to indicate if the load impedance is purely resistive.
    time_varying : bool
        Flag to indicate if the load impedance is time-varying.
    """

    def __init__(self, purely_resistive: bool, time_varying: bool):
        super().__init__()
        self.purely_resistive: bool = purely_resistive
        """Flag to indicate if the load impedance is purely resistive."""
        self.time_varying: bool = time_varying
        """Flag to indicate if the load impedance is time-varying."""


class PurelyResistiveBaseLoad(BaseLoad):
    """Abstract class for a resistance, that can vary in time.

    Parameters
    ----------
    time_varying : bool
        Flag to indicate if the resistance impedance is time-varying.
    """

    def __init__(self, time_varying: bool):
        super().__init__(
            purely_resistive=True,
            time_varying=time_varying,
        )

    @abstractmethod
    def load_impedance(self, t: float) -> float:
        r"""Impedance of the resistance.

        The resistance impedance can depend on the time.
        This function should be implemented in the derived class.

        Parameters
        ----------
        t : float
            Time in seconds.

        Return
        ------
        float
            Load impedance in Ohm. Can vary in time if the resistance is
            time-varying.

        Notes
        -----
        For a constant resistance, the impedance is the resistance

        .. math::

            Z_l = R_l

        where:

        - :math:`R_l` is the resistance in Ohm.


        For a time-varying resistance, the impedance can be defined as:

        .. math::

            Z_l(t) = R_l(t)

        where:

        - :math:`R_l(t)` is the load resistance in Ohm,
          which can vary in time.
        """
        raise NotImplementedError(
            "Define the load impedance calculation in the derived class."
        )


class ComplexImpedanceBaseLoad(BaseLoad):
    """Abstract class for a constant impedance.

    The impedance can be composed of resistance, inductance and capacitance,
    and thus can be complex and depend on the frequency.

    Parameters
    ----------
    purely_resistive : bool
        Flag to indicate if the load impedance is purely resistive.
    """

    def __init__(self, purely_resistive: bool):
        super().__init__(
            purely_resistive=purely_resistive,
            time_varying=False,
        )

    @abstractmethod
    def load_impedance(self, frequency: np.ndarray) -> np.ndarray:
        r"""Impedance of the load.

        The load impedance can depend on the frequency.
        This function should be implemented in the derived class.

        Parameters
        ----------
        frequency : numpy.ndarray
            Array of frequency in Hz.

        Return
        ------
        numpy.ndarray
            Array of load impedance in Ohm. Can be complex if the load is a
            capacitor or an inductor. The impedance is the same for all
            frequencies if the load is a resistance.

        Notes
        -----
        For a constant resistance, the impedance is the resistance

        .. math::

            Z_l = R_l

        where:

        - :math:`R_l` is the load resistance in Ohm.


        For a capacitor, the impedance is:

        .. math::

            Z_l(f) = \frac{1}{j 2 \pi f C_l}

        where:

        - :math:`j` is the imaginary unit,
        - :math:`f` is the frequency in Hz,
        - :math:`C_l` is the load capacitance in Farad.


        For an inductor, the impedance is:

        .. math::

            Z_l(f) = j 2 \pi f L_l

        where:

        - :math:`L_l` is the load inductance in Henry.
        """
        self.check_frequency(frequency)

        raise NotImplementedError(
            "Define the load impedance calculation in the derived class."
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
