import numpy as np

from pyresiflex.load.base_load import PurelyResistiveBaseLoad


class ConstantResistance(PurelyResistiveBaseLoad):
    """Constant resistance load.

    Parameters
    ----------
    R : float
        Load resistance in Ohm.

    Examples
    --------
    >>> from pyresiflex.load.time_varying_resistance import (
    ...     ConstantResistance,
    ... )
    >>> load = ConstantResistance(R=50.0)
    >>> round(float(load.load_impedance(t=1e-6)), 3)
    50.0

    .. minigallery:: pyresiflex.load.time_varying_resistance.ConstantResistance
    """

    def __init__(self, R: float):
        super().__init__(time_varying=False)
        self.R: float = R

    def load_impedance(
        self,
        t: float,
    ) -> float:
        """Load impedance.

        The load impedance is constant and equal to the resistance.

        Parameters
        ----------
        t : float
            Time in seconds. Not used for constant resistance.

        Returns
        -------
        float
            Load impedance in Ohm.

        Notes
        -----
        For a constant resistance, the impedance is the same for all
        frequencies, therefore:

        .. math::

            Z_l = R_l

        where:

        - :math:`Z_l` is the load impedance in Ohm,
        - :math:`R_l` is the load resistance in Ohm.
        """
        return self.R


class PlasmaResistanceLinearFall(PurelyResistiveBaseLoad):
    """Plasma resistance load.

    It is a time-varying resistance, characterized by a linear ramp
    from Z_start to Z_end between t_start_fall and t_end_fall.

    Parameters
    ----------
    Z_start : float
        Start impedance in Ohm.
    Z_end : float
        End impedance in Ohm.
    t_start_fall : float
        Start fall time in second.
    t_end_fall : float
        End fall time in second.

    Examples
    --------
    >>> from pyresiflex.load.time_varying_resistance import (
    ...     PlasmaResistanceLinearFall,
    ... )
    >>> load = PlasmaResistanceLinearFall(
    ...     Z_start=100.0, Z_end=0.0, t_start_fall=0.0, t_end_fall=1.0
    ... )
    >>> round(float(load.load_impedance(t=0.5)), 3)
    50.0

    .. minigallery::
       pyresiflex.load.time_varying_resistance.PlasmaResistanceLinearFall
    """

    def __init__(
        self,
        Z_start: float,
        Z_end: float,
        t_start_fall: float,
        t_end_fall: float,
    ):
        super().__init__(time_varying=True)

        self.Z_start: float = Z_start
        self.Z_end: float = Z_end
        self.t_start_fall: float = t_start_fall
        self.t_end_fall: float = t_end_fall

    def load_impedance(
        self,
        t: float,
    ) -> float:
        r"""Load impedance.

        The load impedance is a linear ramp from Z_start to Z_end
        between t_start_fall and t_end_fall.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Load impedance in Ohm.

        Notes
        -----
        The impedance of a plasma resistance is defined as:

        .. math::

            Z_l(t) = Z_{start} + (Z_{end} - Z_{start})
                    \frac{t - t_{start}}{t_{end} - t_{start}}

            Z_l(t) =
                \begin{cases}
                Z_{start}, \text{if } t < t_{start} \\
                Z_{start} + (Z_{end} - Z_{start})
                    \frac{t - t_{start}}{t_{end} - t_{start}},
                    \text{if } t_{start} < t < t_{end} \\
                Z_{end}, \text{if } t > t_{end}
                \end{cases}

        where:

        - :math:`Z_l(t)` is the load impedance at time :math:`t`,
        - :math:`Z_{start}` is the start impedance,
        - :math:`Z_{end}` is the end impedance,
        - :math:`t_{start}` is the start fall time,
        - :math:`t_{end}` is the end fall time.
        """
        if t < self.t_start_fall:
            return self.Z_start
        if t < self.t_end_fall:
            # Linear ramp from Z_start to Z_end.
            return self.Z_start + (self.Z_end - self.Z_start) * (
                t - self.t_start_fall
            ) / (self.t_end_fall - self.t_start_fall)
        return self.Z_end


class PlasmaResistanceExponentialFall(PurelyResistiveBaseLoad):
    """Plasma resistance load.

    It is a time-varying resistance, characterized by an exponential fall
    from Z_start to Z_end between t_start_fall and t_end_fall.

    Parameters
    ----------
    Z_start : float
        Start impedance in Ohm.
    Z_end : float
        End impedance in Ohm.
    t_start_fall : float
        Start fall time in second.
    t_end_fall : float
        End fall time in second.

    Examples
    --------
    >>> from pyresiflex.load.time_varying_resistance import (
    ...     PlasmaResistanceExponentialFall,
    ... )
    >>> load = PlasmaResistanceExponentialFall(
    ...     Z_start=100.0, Z_end=0.0, t_start_fall=0.0, t_end_fall=1.0
    ... )
    >>> round(float(load.load_impedance(t=0.5)), 3)
    50.0

    .. minigallery::
       pyresiflex.load.time_varying_resistance.PlasmaResistanceExponentialFall
    """

    def __init__(
        self,
        Z_start: float,
        Z_end: float,
        t_start_fall: float,
        t_end_fall: float,
        fall_exponent: float = 1.0,
    ):
        super().__init__(time_varying=True)

        self.Z_start: float = Z_start
        self.Z_end: float = Z_end
        self.t_start_fall: float = t_start_fall
        self.t_end_fall: float = t_end_fall
        self.fall_exponent: float = fall_exponent

    def load_impedance(
        self,
        t: float,
    ) -> float:
        r"""Load impedance.

        The load impedance is an exponential fall from Z_start to Z_end
        between t_start_fall and t_end_fall.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Load impedance in Ohm.

        """
        if t < self.t_start_fall:
            return self.Z_start
        if t < self.t_end_fall:
            # Exponential fall from Z_start to Z_end.
            return (
                self.Z_end
                + (self.Z_start - self.Z_end)
                * (
                    1
                    - (t - self.t_start_fall)
                    / (self.t_end_fall - self.t_start_fall)
                )
                ** self.fall_exponent
            )
        return self.Z_end


class PlasmaResistanceInterpolate(PurelyResistiveBaseLoad):
    """Plasma resistance load.

    It is a time-varying resistance, characterized by interpolation
    between given time points.

    Parameters
    ----------
    t_array : numpy.ndarray
        List of time points in second.
    R_array : numpy.ndarray
        List of resistance points in Ohm.
        Must have the same length as t_array.

    Examples
    --------
    >>> import numpy as np
    >>> from pyresiflex.load.time_varying_resistance import (
    ...     PlasmaResistanceInterpolate,
    ... )
    >>> t_array = np.array([0.0, 1.0])
    >>> R_array = np.array([100.0, 0.0])
    >>> load = PlasmaResistanceInterpolate(t_array, R_array)
    >>> load.check_arrays(t_array, R_array) is None
    True
    >>> round(float(load.load_impedance(t=0.5)), 3)
    50.0

    .. minigallery::
       pyresiflex.load.time_varying_resistance.PlasmaResistanceInterpolate
    """

    def __init__(self, t_array: np.ndarray, R_array: np.ndarray):
        super().__init__(time_varying=True)
        self.check_arrays(t_array, R_array)
        self.t_array: np.ndarray = np.array(t_array)
        self.R_array: np.ndarray = np.array(R_array)

    def check_arrays(self, t_array: np.ndarray, R_array: np.ndarray):
        """Check that the input arrays are valid.

        Parameters
        ----------
        t_array : numpy.ndarray
            List of time points in second.
        R_array : numpy.ndarray
            List of resistance points in Ohm.

        Raises
        ------
        ValueError
            If the input arrays do not have the same length or
            if they have less than 2 points.
        """
        # Check that the input arrays are numpy arrays.
        if not isinstance(t_array, np.ndarray):
            raise TypeError("t_array must be a numpy array.")
        if not isinstance(R_array, np.ndarray):
            raise TypeError("R_array must be a numpy array.")
        # Check that the input arrays are one-dimensional.
        if t_array.ndim != 1:
            raise ValueError("t_array must be one-dimensional.")
        if R_array.ndim != 1:
            raise ValueError("R_array must be one-dimensional.")
        # Check that the input arrays contain real values.
        if not np.isrealobj(t_array):
            raise ValueError("t_array must contain real values.")
        if not np.isrealobj(R_array):
            raise ValueError("R_array must contain real values.")
        # Check that the input arrays contain finite values.
        if not np.all(np.isfinite(t_array)):
            raise ValueError("t_array must contain finite values.")
        if not np.all(np.isfinite(R_array)):
            raise ValueError("R_array must contain finite values.")
        # Check that the input arrays have the same length.
        if len(t_array) != len(R_array):
            raise ValueError("t_array and R_array must have the same length.")
        # Check that the input arrays have at least 2 points.
        if len(t_array) < 2:
            raise ValueError(
                "t_array and R_array must have at least 2 points."
            )
        # Check that the time array is sorted in ascending order.
        if not np.all(np.diff(t_array) > 0):
            raise ValueError("t_array must be sorted in ascending order.")

    def load_impedance(
        self,
        t: float,
    ) -> float:
        """Load impedance.

        The load impedance is interpolated between the given time points.
        Note that values outside the interpolation range will be clamped.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Load impedance in Ohm.
        """
        return np.interp(
            t,
            self.t_array,
            self.R_array,
            left=self.R_array[0],
            right=self.R_array[-1],
        )
