from typing import Callable
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.base_generator import PurelyResistiveBaseGenerator
from pyresiflex.load.time_varying_resistance import PlasmaResistanceInterpolate
from pyresiflex.misc.plot import plot_voltage_current


class PurelyResistiveExperiment:
    """Analyze experimental data from a purely resistive experiment.

    This class allows to analyze experimental data from a purely resistive
    experiment. It can compute the plasma resistance from measured voltage and
    current signals.

    Parameters
    ----------
    experimental_voltage_time : numpy.ndarray
        Time array of the measured voltage signal in second.
    experimental_voltage_value : numpy.ndarray
        Measured voltage signal in Volt.
    x_meas_voltage : float
        Position of the voltage measurement from the generator in meter.
    experimental_current_time : numpy.ndarray
        Time array of the measured current signal in second.
    experimental_current_value : numpy.ndarray
        Measured current signal in Ampere.
    x_meas_current : float
        Position of the current measurement from the generator in meter.
    cable : PerfectCable, optional
        Perfect transmission line object, by default None.
        If None, `L`, `Z_c` and `c` must be provided.
    L : float, optional
        Length of the cable in meter, by default None.
        If `cable` is None, `L`, `Z_c` and `c` must be provided.
    Z_c : float, optional
        Characteristic impedance of the cable in Ohm, by default None.
        If `cable` is None, `L`, `Z_c` and `c` must be provided.
    c : float, optional
        Speed of light in vacuum in meter per second, by default None.
        If `cable` is None, `L`, `Z_c` and `c` must be provided.
    correct_time_zero : bool, optional
        If True, the time is corrected to account for the propagation delay
        from the generator to the measurement positions, by default True.
        Time is then referenced to the generator position.
    """

    def __init__(
        self,
        experimental_voltage_time: np.ndarray,
        experimental_voltage_value: np.ndarray,
        x_meas_voltage: float,
        experimental_current_time: np.ndarray,
        experimental_current_value: np.ndarray,
        x_meas_current: float,
        cable: PerfectCable | None = None,
        L: float | None = None,
        Z_c: float | None = None,
        c: float | None = None,
        correct_time_zero: bool = True,
    ):
        self.experimental_voltage_time = experimental_voltage_time
        self.experimental_voltage_value = experimental_voltage_value
        self.experimental_current_time = experimental_current_time
        self.experimental_current_value = experimental_current_value

        # Get or create the cable object.
        self.cable: PerfectCable
        if cable is not None:
            if not (isinstance(cable, PerfectCable)):
                raise TypeError(
                    "cable must be an instance of PerfectCable or None"
                )
            if (L is not None) or (Z_c is not None) or (c is not None):
                raise ValueError(
                    "If cable is not None, L, Z_c and c must be None"
                )
            self.cable = cable
        elif (L is not None) and (Z_c is not None) and (c is not None):
            self.cable = PerfectCable(L, Z_c, c)
        else:
            raise ValueError(
                "Either `cable`, or all of `L`, `Z_c` and `c`,"
                " must be provided."
            )

        self.x_meas_voltage = x_meas_voltage
        self.x_meas_current = x_meas_current

        self.correct_time_zero = correct_time_zero

        # To be filled when computing the resistance.
        self.times: np.ndarray
        self.R_p: np.ndarray
        self.times_corrected: np.ndarray
        self.Rp_corrected: np.ndarray
        self.times_corrected_with_nan: np.ndarray
        self.Rp_corrected_with_nan: np.ndarray
        self.load: PlasmaResistanceInterpolate
        self.load_corrected: PlasmaResistanceInterpolate

    def V_meas(self, t: np.ndarray | float) -> np.ndarray:
        """Interpolate measured voltage signal at position `x_meas_voltage`.

        If `correct_time_zero` is True, the time is corrected to account for
        the propagation delay from the generator to the voltage measurement.

        Parameters
        ----------
        t : numpy.ndarray or float
            Time array in second.

        Returns
        -------
        numpy.ndarray
            Interpolated voltage signal in Volt.
        """
        if self.correct_time_zero:
            # Correct time to account for propagation delay.
            # Origin of time is at the generator.
            t = t - self.x_meas_voltage / self.cable.c
        return np.interp(
            np.asarray(t),
            self.experimental_voltage_time,
            self.experimental_voltage_value,
            left=0,
            right=0,
        )

    def I_meas(self, t: np.ndarray | float) -> np.ndarray:
        """Interpolate measured current signal at position `x_meas_current`.

        If `correct_time_zero` is True, the time is corrected to account for
        the propagation delay from the generator to the current measurement.

        Parameters
        ----------
        t : numpy.ndarray or float
            Time array in second.

        Returns
        -------
        numpy.ndarray
            Interpolated current signal in Ampere.
        """
        if self.correct_time_zero:
            # Correct time to account for propagation delay.
            # Origin of time is at the generator.
            t = t - self.x_meas_current / self.cable.c
        return np.interp(
            np.asarray(t),
            self.experimental_current_time,
            self.experimental_current_value,
            left=0,
            right=0,
        )

    def compute_plasma_resistance_from_vmeas_and_imeas(
        self,
        times: np.ndarray,
        threshold: float = 0.0,
        channel_formation_time: float = 0.0,
        interpolate_with_previous_value: bool = False,
    ) -> np.ndarray:
        r"""Compute the plasma resistance from measured voltage and current.

        Parameters
        ----------
        times : numpy.ndarray
            Time array in second.
        threshold : float, optional
            Threshold to avoid division by zero, by default 0.0.
            When the denominator is smaller than this threshold, the resistance
            is set to nan.
        channel_formation_time : float, optional
            Time in second before which the resistance is not correctly
            computed, by default 0.0.
            This is because the reflected wave has not yet reached the
            measurement position.
        interpolate_with_previous_value : bool, optional
            If True, when there are nan values in the resistance, they are
            replaced by the previous valid value for interpolation. This allows
            to keep the same time array for interpolation. If False, nan values
            are removed from the resistance and time arrays for interpolation.
            By default, False.

        Returns
        -------
        numpy.ndarray
            Plasma resistance in Ohm.

        Notes
        -----
        This method assumes that the measurements are done at the same position
        in the cable, i.e. `x_meas_voltage` must be equal to `x_meas_current`.

        Then, the plasma resistance is computed using the following formula:

        .. math::

            R_p(t) = Z_c \frac{
                \left[V_{meas}(t - \tau) + Z_c I_{meas}(t - \tau)\right]
                +
                \left[V_{meas}(t + \tau) - Z_c I_{meas}(t + \tau)\right]
                }{
                \left[V_{meas}(t - \tau) + Z_c I_{meas}(t - \tau)\right]
                -
                \left[V_{meas}(t + \tau) - Z_c I_{meas}(t + \tau)\right]
                }

        where:

        - :math:`R_p` is the plasma resistance in Ohm,
        - :math:`Z_c` is the characteristic impedance of the cable in Ohm,
        - :math:`V` is the voltage in Volt,
        - :math:`I` is the current in Ampere,
        - :math:`x_{meas}` is the position of the measurements along the cable
          in meter,
        - :math:`L` is the length of the cable in meter,
        - :math:`c` is the wave velocity in the cable in m/s,
        - :math:`\tau = \frac{L - x_{meas}}{c}` is the propagation delay from
          the measurement position to the end of the cable in second.
        """
        if not self.x_meas_voltage == self.x_meas_current:
            raise ValueError(
                "`x_meas_voltage` must be equal to `x_meas_current` to compute"
                " the plasma resistance using this method."
            )
        x_meas = self.x_meas_voltage

        # Get cable parameters.
        Z_c = self.cable.Z_c
        L = self.cable.L
        c = self.cable.c
        # Define propagation delay from measurement position to the end of
        # the cable.
        tau = (L - x_meas) / c

        # Compute incident and reflected voltages at the measurement position.
        V_i = self.V_meas(times - tau) + Z_c * self.I_meas(times - tau)
        V_r = self.V_meas(times + tau) - Z_c * self.I_meas(times + tau)

        # Compute plasma resistance.
        numerator = V_i + V_r
        denominator = V_i - V_r
        # If there is a division by zero, set the resistance to a high value.
        if np.any(denominator == 0):
            warn(
                "Some values in the denominator are zero, "
                "resistance cannot be computed correctly."
                " These values are set to 1 MOhm."
            )
            R_p = np.full_like(denominator, 1e6)
            mask = denominator != 0
            R_p[mask] = Z_c * numerator[mask] / denominator[mask]
        else:
            R_p = Z_c * numerator / denominator

        # Save time and resistance.
        self.times = times
        self.R_p = R_p

        # Also save a corrected resistance, where values are set to nan when:
        Rp_corrected = np.copy(R_p)
        # - the denominator is too small,
        Rp_corrected[np.abs(denominator) < threshold] = np.nan
        # - the resistance is negative.
        Rp_corrected[Rp_corrected < 0] = np.nan
        # For the time before `channel_formation_time`, the resistance is not
        # correctly computed, because the reflected wave has not yet reached
        # the measurement position. Set these values to a high value (1 MΩ).
        Rp_corrected[self.times < channel_formation_time] = 1e6
        # - Remove nan values from the resistance for interpolation.
        self.times_corrected = np.copy(self.times)
        if not interpolate_with_previous_value:
            self.times_corrected = self.times_corrected[
                ~np.isnan(Rp_corrected)
            ]

        # Save versions with nan for plotting.
        self.times_corrected_with_nan = np.copy(self.times)
        self.times_corrected_with_nan[np.isnan(Rp_corrected)] = np.nan
        self.Rp_corrected_with_nan = np.copy(Rp_corrected)
        self.Rp_corrected_with_nan[denominator == 0] = np.nan

        # Save corrected resistance without nan values.
        self.Rp_corrected = np.copy(Rp_corrected)
        if interpolate_with_previous_value:
            # Instead of removing nan values, set them to the previous valid
            # value for interpolation.
            for i in range(1, len(self.Rp_corrected)):
                if np.isnan(self.Rp_corrected[i]):
                    self.Rp_corrected[i] = self.Rp_corrected[i - 1]
        else:
            # Remove nan values from the resistance for interpolation.
            self.Rp_corrected = self.Rp_corrected[~np.isnan(self.Rp_corrected)]

        # Create a time-varying load from the resistance.
        self.load = PlasmaResistanceInterpolate(self.times, self.R_p)
        # Create a time-varying load from the corrected resistance.
        self.load_corrected = PlasmaResistanceInterpolate(
            self.times_corrected, self.Rp_corrected
        )

        return self.R_p

    def compute_plasma_resistance_from_vmeas_and_vg(
        self,
        times: np.ndarray,
        generator: PurelyResistiveBaseGenerator,
        max_n: int = 5,
    ) -> np.ndarray:
        r"""Compute the plasma resistance from measured voltage and generator.

        Parameters
        ----------
        times : numpy.ndarray
            Time array in second.
        generator : PurelyResistiveBaseGenerator
            Purely resistive generator object.
        max_n : int, optional
            Maximum number of wave generations to reconstruct, by default 5.

        Returns
        -------
        numpy.ndarray
            Plasma resistance in Ohm.
        """
        return self._compute_plasma_resistance_from_vg(
            times,
            generator,
            max_n=max_n,
            x_meas=self.x_meas_voltage,
            lead=lambda s: float(self.V_meas(s)),
            sign=-1.0,
        )

    def compute_plasma_resistance_from_imeas_and_vg(
        self,
        times: np.ndarray,
        generator: PurelyResistiveBaseGenerator,
        max_n: int = 5,
    ) -> np.ndarray:
        r"""Compute the plasma resistance from measured current and generator.

        Parameters
        ----------
        times : numpy.ndarray
            Time array in second.
        generator : PurelyResistiveBaseGenerator
            Purely resistive generator object.
        max_n : int, optional
            Maximum number of wave generations to reconstruct, by default 5.

        Returns
        -------
        numpy.ndarray
            Plasma resistance in Ohm.
        """
        Z_c = self.cable.Z_c
        return self._compute_plasma_resistance_from_vg(
            times,
            generator,
            max_n=max_n,
            x_meas=self.x_meas_current,
            lead=lambda s: -Z_c * float(self.I_meas(s)),
            sign=1.0,
        )

    def _compute_plasma_resistance_from_vg(
        self,
        times: np.ndarray,
        generator: PurelyResistiveBaseGenerator,
        max_n: int,
        x_meas: float,
        lead: Callable[[float], float],
        sign: float,
    ) -> np.ndarray:
        """Reconstruct the plasma resistance from the generator voltage.

        Shared implementation behind
        :meth:`compute_plasma_resistance_from_vmeas_and_vg` and
        :meth:`compute_plasma_resistance_from_imeas_and_vg`, which only
        differ by the measured signal used to seed the reconstruction.

        Parameters
        ----------
        times : numpy.ndarray
            Time array in second.
        generator : PurelyResistiveBaseGenerator
            Purely resistive generator object.
        max_n : int
            Maximum number of wave generations to reconstruct.
        x_meas : float
            Position of the measurement in meters.
        lead : Callable
            Leading measured term as a function of ``t + tau``.
        sign : float
            Sign of the reconstructed series in the numerator.

        Returns
        -------
        numpy.ndarray
            Plasma resistance in Ohm.
        """
        if not isinstance(generator, PurelyResistiveBaseGenerator):
            raise TypeError(
                "generator must be an instance of PurelyResistiveBaseGenerator"
            )
        # Get cable properties.
        Z_c = self.cable.Z_c
        # Get generator properties.
        self.generator = generator
        R_g = self.generator.R_g
        # Define attenuation coefficient at the generator.
        self.alpha_g = Z_c / (Z_c + R_g)
        # Define reflection coefficient at the generator.
        self.gamma_g = (R_g - Z_c) / (R_g + Z_c)

        reconstructed_resistance = np.zeros_like(times)
        # `gamma_p(t)` is a pure function of `t`, so the (recursively
        # computed) reflection coefficients are cached and reused both
        # within a time point and across all the time points.
        memo: dict[float, float] = {}
        for i, t in enumerate(times):
            # Compute reflection coefficient at the load.
            gamma_p = self._reconstruct_gamma_load(
                t, x_meas, lead, sign, max_n, memo
            )
            # Compute plasma resistance from reflection coefficient.
            reconstructed_resistance[i] = self.get_resistance_from_gamma(
                Z_c=Z_c, gamma_l=gamma_p
            )

        return reconstructed_resistance

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

    def get_gamma_load_from_measured_voltage(
        self,
        t: float,
        max_n: int,
    ) -> float:
        r"""Get the reflection coefficient at the load from a measured voltage.

        Parameters
        ----------
        t : float
            Time in seconds.
        max_n : int
            Maximum number of generations to consider.
            To be removed once the computation is optimized.

        Returns
        -------
        float
            Reflection coefficient at the load.

        Notes
        -----
        To get the reflection coefficient at the load, we need to rebuild
        the incident and reflected waves, recursively, using:

        .. math::

            \Gamma_p(t)
            =
            \frac{
                V_{\text{meas}}\left(t+\tau\right)
                -\sum_{n=0}^N(t+\tau - \frac{x_{\text{meas}}}{c})
                    \alpha_g
                    V_g \left(
                        t
                        + \tau
                        - \frac{x_{\text{meas}}}{c}
                        - \frac{2 n L}{c}
                    \right)
                    \prod_{k=1}^n \left[
                        \Gamma_g
                        \Gamma_p \left(
                            t
                            + \tau
                            - \frac{x_{\text{meas}}}{c}
                            - \frac{(2 k-1) L}{c}
                        \right)
                    \right]
                }{
                \sum_{n=0}^{N(t+\tau + \frac{x_{\text{meas}}}{c} - 2 L / c)}
                    \alpha_g
                    V_g\left(
                        t
                        + \tau
                        + \frac{x_{\text{meas}}}{c}
                        - \frac{2(n+1) L}{c}
                    \right)
                    \prod_{k=1}^n \left[
                        \Gamma_g
                        \Gamma_p \left(
                            t
                            + \tau
                            + \frac{x_{\text{meas}}}{c}
                            - \frac{2 (k+1) L}{c}
                        \right)
                    \right]
                }
        """
        return self._reconstruct_gamma_load(
            t,
            x_meas=self.x_meas_voltage,
            lead=lambda s: float(self.V_meas(s)),
            sign=-1.0,
            max_n=max_n,
            memo={},
        )

    def get_gamma_load_from_measured_current(
        self,
        t: float,
        max_n: int,
    ) -> float:
        r"""Get the reflection coefficient at the load from a measured current.

        Parameters
        ----------
        t : float
            Time in seconds.
        max_n : int
            Maximum number of generations to consider.
            To be removed once the computation is optimized.

        Returns
        -------
        float
            Reflection coefficient at the load.

        Notes
        -----
        To get the reflection coefficient at the load, we need to rebuild
        the incident and reflected waves, recursively, using:

        .. math::

            \Gamma_p(t)
            =
            \frac{
                -Z_c I_{\text{meas}}\left(t+\tau\right)
                +\sum_{n=0}^N(t+\tau - \frac{x_{\text{meas}}}{c})
                    \alpha_g
                    V_g \left(
                        t
                        + \tau
                        - \frac{x_{\text{meas}}}{c}
                        - \frac{2 n L}{c}
                    \right)
                    \prod_{k=1}^n \left[
                        \Gamma_g
                        \Gamma_p \left(
                            t
                            + \tau
                            - \frac{x_{\text{meas}}}{c}
                            - \frac{(2 k-1) L}{c}
                        \right)
                    \right]
                }{
                \sum_{n=0}^{N(t+\tau + \frac{x_{\text{meas}}}{c} - 2 L / c)}
                    \alpha_g
                    V_g\left(
                        t
                        + \tau
                        + \frac{x_{\text{meas}}}{c}
                        - \frac{2(n+1) L}{c}
                    \right)
                    \prod_{k=1}^n \left[
                        \Gamma_g
                        \Gamma_p \left(
                            t
                            + \tau
                            + \frac{x_{\text{meas}}}{c}
                            - \frac{2 (k+1) L}{c}
                        \right)
                    \right]
                }
        """
        Z_c = self.cable.Z_c
        return self._reconstruct_gamma_load(
            t,
            x_meas=self.x_meas_current,
            lead=lambda s: -Z_c * float(self.I_meas(s)),
            sign=1.0,
            max_n=max_n,
            memo={},
        )

    def _reconstruct_gamma_load(
        self,
        t: float,
        x_meas: float,
        lead: Callable[[float], float],
        sign: float,
        max_n: int,
        memo: dict[float, float],
    ) -> float:
        r"""Recursively reconstruct the load reflection coefficient at time t.

        Shared, memoized implementation behind
        :meth:`get_gamma_load_from_measured_voltage` and
        :meth:`get_gamma_load_from_measured_current`, which only differ by the
        leading measured term (`lead`) and the sign of the reconstructed
        series (`sign`).

        ``Gamma_p(t)`` depends only on ``Gamma_p`` at strictly earlier times,
        so the reconstructed reflection coefficients are cached in `memo`:
        each distinct time is computed only once. This turns the naive
        (exponential) recursion into a computation that scales linearly with
        the number of distinct times reached.

        Parameters
        ----------
        t : float
            Time in seconds.
        x_meas : float
            Position of the measurement in meters.
        lead : Callable
            Leading measured term as a function of ``t + tau`` (``V_meas``
            for a voltage measurement, ``-Z_c * I_meas`` for a current one).
        sign : float
            Sign of the reconstructed series in the numerator (``-1`` for a
            voltage measurement, ``+1`` for a current one).
        max_n : int
            Maximum number of wave generations to reconstruct.
        memo : dict
            Cache mapping a time to its reconstructed reflection coefficient.

        Returns
        -------
        float
            Reflection coefficient at the load.
        """
        if t in memo:
            return memo[t]

        L = self.cable.L
        c = self.cable.c

        if x_meas > L or x_meas < 0:
            raise ValueError("x_meas must be between 0 and L")

        # Before the reflected wave returns to the measurement point, the
        # reflection coefficient cannot be accessed.
        if t <= L / c:
            memo[t] = np.nan
            return np.nan

        # Reflection coefficient at the generator is supposed to be constant
        # and purely resistive, and the generator voltage perfectly known.
        gamma_g = self.gamma_g
        alpha_g = self.alpha_g
        V_g = self.generator.generator_voltage

        tau = (L - x_meas) / c

        N_incident = self.N(t + tau - x_meas / c)
        N_reflected = self.N(t + tau + x_meas / c - 2 * L / c)

        if N_incident > max_n or N_reflected > max_n:
            memo[t] = np.nan
            return np.nan

        # Numerator: measured signal corrected by the reconstructed series.
        num = lead(t + tau)
        for n in range(0, N_incident + 1):
            term = alpha_g * V_g(t + tau - x_meas / c - 2 * n * L / c)
            for k in range(1, n + 1):
                gamma_p = self._reconstruct_gamma_load(
                    t + tau - x_meas / c - (2 * k - 1) * L / c,
                    x_meas,
                    lead,
                    sign,
                    max_n,
                    memo,
                )
                term *= gamma_p * gamma_g
            num += sign * term

        # Denominator: reconstructed reflected series.
        denom = 0.0
        for n in range(0, N_reflected + 1):
            term = alpha_g * V_g(t + tau + x_meas / c - 2 * (n + 1) * L / c)
            for k in range(1, n + 1):
                gamma_p = self._reconstruct_gamma_load(
                    t + tau + x_meas / c - (2 * k + 1) * L / c,
                    x_meas,
                    lead,
                    sign,
                    max_n,
                    memo,
                )
                term *= gamma_p * gamma_g
            denom += term

        result = np.nan if denom == 0 else num / denom
        memo[t] = result
        return result

    @staticmethod
    def get_resistance_from_gamma(Z_c: float, gamma_l: float) -> float:
        r"""Get the load impedance from the reflection coefficient.

        Parameters
        ----------
        Z_c : float
            Characteristic impedance of the cable in Ohm.
        gamma_l : float
            Reflection coefficient at the load.

        Returns
        -------
        float
            Load impedance in Ohm.

        Notes
        -----
        The load impedance is defined as:

        .. math::

            Z_l = Z_c \frac{1 + \Gamma_l}{1 - \Gamma_l}

        where:

        - :math:`Z_l` is the load impedance in Ohm,
        - :math:`Z_c` is the characteristic impedance of the cable in Ohm,
        - :math:`\Gamma_l` is the reflection coefficient at the load.
        """
        return Z_c * (1 + gamma_l) / (1 - gamma_l)

    def plot_voltage_and_current(
        self,
    ) -> tuple[Figure, Axes, Axes]:
        """Plot the measured voltage and current signals.

        The time axis is corrected to account for the propagation delay
        from the generator to the measurement positions if
        `correct_time_zero` is True.

        Returns
        -------
        tuple of Figure, Axes, Axes
            Figure and axes objects of the voltage and current plots.
        """
        # Compute the time offsets to correct the time axis.
        voltage_time_offset = 0.0
        current_time_offset = 0.0
        if self.correct_time_zero:
            voltage_time_offset = self.x_meas_voltage / self.cable.c
            current_time_offset = self.x_meas_current / self.cable.c

        fig, ax_v, ax_i = plot_voltage_current(
            voltage_time=self.experimental_voltage_time + voltage_time_offset,
            voltage_value=self.experimental_voltage_value,
            current_time=self.experimental_current_time + current_time_offset,
            current_value=self.experimental_current_value,
        )
        return fig, ax_v, ax_i

    def plot_resistance(
        self,
        times: np.ndarray,
        plot_whole: bool = False,
        plot_corrected: bool = True,
        plot_interpolated: bool = True,
        _also_plot_when_near_cable_impedance: bool = True,
        show: bool = False,
        legend: bool = True,
        figax: tuple[Figure, Axes] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot the computed plasma resistance.

        Parameters
        ----------
        times : numpy.ndarray
            Time array in second.
        plot_whole : bool, optional
            If True, the whole computed resistance is plotted in light gray.
            By default, False.
        plot_corrected : bool, optional
            If True, the corrected resistance is plotted in full black line.
            By default, True.
        plot_interpolated : bool, optional
            If True, the interpolated resistance is also plotted.
            By default, True
        _also_plot_when_near_cable_impedance : bool, optional
            If True, resistance values near the cable impedance are also
            plotted. By default, True.
            When there is no more reflected wave, the resistance tends to the
            cable impedance. This option allows to hide these values for
            better visualization of the plasma resistance evolution.
        show : bool, optional
            If True, plot is shown, by default False.
        legend : bool, optional
            If True, legend is added to the plot, by default True.
        figax : tuple of Figure, Axes, optional
            Figure and axes objects to use for the plot, by default None.

        Returns
        -------
        tuple of Figure, Axes
            Figure and axes objects of the resistance plot.
        """
        # Check that the resistance has been computed.
        if not hasattr(self, "R_p"):
            raise ValueError(
                "The resistance has not been computed yet. "
                "Please call compute_plasma_resistance_from_vmeas_and_imeas() "
                "first."
            )

        # Create figure and axes if not provided.
        if figax is not None:
            fig, ax = figax
        else:
            fig, ax = plt.subplots()

        # Plot the whole resistance in light gray.
        if plot_whole:
            R_p = [self.load.load_impedance(t) for t in times]
            ax.plot(times * 1e9, R_p, color="lightgray")

        # Plot corrected resistance values in full black line.
        if plot_corrected:
            t = self.times_corrected_with_nan * 1e9
            R = self.Rp_corrected_with_nan
            # Optionally hide values near the cable impedance.
            if not _also_plot_when_near_cable_impedance:
                Z_c = self.cable.Z_c
                tol = 0.001 * Z_c  # 0.1% tolerance
                mask = np.abs(self.Rp_corrected_with_nan - Z_c) < tol
                t = self.times_corrected_with_nan[~mask] * 1e9
                R = self.Rp_corrected_with_nan[~mask]

            ax.plot(
                t,
                R,
                color="k",
                ls="-",
            )

        # Plot interpolated resistance in dashed black line.
        if plot_interpolated:
            R_p_corrected = [
                self.load_corrected.load_impedance(t) for t in times
            ]
            ax.plot(
                times * 1e9,
                R_p_corrected,
                color="k",
                ls="--",
                lw=2,
            )

        # Plot settings.
        ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
        ax.set_ylabel(r"$\mathregular{R_p \, [\Omega]}$")
        ax.set_xlim(times[0] * 1e9, times[-1] * 1e9)

        # Set y-limits to avoid inf values.
        y_lim_min = np.nanmin(self.R_p)
        if np.abs(y_lim_min) == np.inf:
            y_lim_min = 0.0
        y_lim_max = np.nanmax(self.R_p)
        if np.abs(y_lim_max) == np.inf:
            y_lim_max = 1.0
        ax.set_ylim(y_lim_min, y_lim_max)

        # Add legend if requested.
        if legend:
            labels = []
            if plot_whole:
                labels.append("Computed resistance")
            if plot_corrected:
                labels.append("Corrected resistance")
            if plot_interpolated:
                labels.append("Interpolated resistance")
            ax.legend(labels)

        # Show the plot if requested.
        if show:
            plt.show()

        return fig, ax
