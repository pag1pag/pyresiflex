import numpy as np

from pyresiflex.load.base_load import ComplexImpedanceBaseLoad


class ConstantResistance(ComplexImpedanceBaseLoad):
    """Constant resistance load.

    Parameters
    ----------
    R : float, optional
        Load resistance in Ohm.
    """

    def __init__(self, R: float):
        super().__init__(purely_resistive=True)
        self.R: float = R

    def load_impedance(
        self,
        frequency: np.ndarray,
    ) -> np.ndarray:
        r"""Load impedance.

        Parameters
        ----------
        frequency : numpy.ndarray
            Frequency array in Hz. Not used for constant resistance.

        Returns
        -------
        numpy.ndarray
            Load impedance in Ohm.

        Notes
        -----
        The impedance of a constant resistance is defined as:

        .. math::

            Z_l(f) = R_l

        where:

        - :math:`Z_l` is the load impedance in Ohm,
        - :math:`R_l` is the load resistance in Ohm.
        """
        self.check_frequency(frequency)
        return self.R * np.ones_like(frequency)


class Capacitance(ComplexImpedanceBaseLoad):
    """Constant capacitance load.

    Parameters
    ----------
    C : float, optional
        Load capacitance in Farad.
    """

    def __init__(self, C: float):
        super().__init__(purely_resistive=False)
        self.C: float = C

    def load_impedance(
        self,
        frequency: np.ndarray,
    ) -> np.ndarray:
        r"""Load impedance.

        Parameters
        ----------
        frequency : numpy.ndarray
            Frequency array in Hz.

        Returns
        -------
        numpy.ndarray
            Load impedance in Ohm.

        Notes
        -----
        The impedance of a capacitance is defined as:

        .. math::

            Z_l(f) = \frac{1}{j 2 \pi f C_l}

        where:

        - :math:`Z_l` is the load impedance in Ohm,
        - :math:`j` is the imaginary unit,
        - :math:`f` is the frequency in Hz,
        - :math:`C_l` is the load capacitance in Farad.

        At zero frequency (DC) the capacitor is an open circuit, so the
        impedance is (positive) infinite.
        """
        self.check_frequency(frequency)
        # At f = 0 the capacitor is an open circuit (infinite impedance);
        # silence the expected divide-by-zero rather than warn on it.
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1 / (1j * 2 * np.pi * frequency * self.C)


class Inductance(ComplexImpedanceBaseLoad):
    """Constant inductance load.

    Parameters
    ----------
    L : float, optional
        Load inductance in Henry.
    """

    def __init__(self, L: float):
        super().__init__(purely_resistive=False)
        self.L: float = L

    def load_impedance(
        self,
        frequency: np.ndarray,
    ) -> np.ndarray:
        r"""Load impedance.

        Parameters
        ----------
        frequency : numpy.ndarray
            Frequency array in Hz.

        Returns
        -------
        numpy.ndarray
            Load impedance in Ohm.

        Notes
        -----
        The impedance of an inductance is defined as:

        .. math::

            Z_l(f) = j 2 \pi f L_l

        where:

        - :math:`Z_l` is the load impedance in Ohm,
        - :math:`j` is the imaginary unit,
        - :math:`f` is the frequency in Hz,
        - :math:`L_l` is the load inductance in Henry.
        """
        self.check_frequency(frequency)
        return 1j * 2 * np.pi * frequency * self.L
