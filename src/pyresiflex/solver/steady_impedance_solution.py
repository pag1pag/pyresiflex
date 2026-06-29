from typing import Callable
from warnings import warn

import numpy as np

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.base_generator import ComplexImpedanceBaseGenerator
from pyresiflex.load.base_load import ComplexImpedanceBaseLoad
from pyresiflex.solver.base_solution import BaseSolution


class SteadyImpedanceSolution(BaseSolution):
    r"""Voltage and current solution for a cable with steady impedance.

    The cable is assumed to be a perfect transmission line, meaning that
    there is no loss in the line and that the wave velocity is constant.
    It is connected to a generator and a load. The generator provides
    a voltage source and a constant impedance, while the load provides
    a constant impedance at the end of the line. Both impedances can be
    complex.

    This class provides the analytical solution for the voltage and current
    in the transmission line, taking into account the generator and load
    impedances. The solution is based on the wave equation and the boundary
    conditions imposed by the generator and load.

    Parameters
    ----------
    cable : PerfectCable
        Perfect transmission line object.
    generator : ComplexImpedanceBaseGenerator
        Generator object that provides the voltage source.
    load : ComplexImpedanceBaseLoad
        Load object that provides the impedance at the end of the line.
    nb_points_ft : int, optional
        Number of points for the Fourier transform of the generator voltage.
        Default is 1000.
        There must be enough points to satisfy the Shannon-Nyquist criterion.
    max_time_ft : float, optional
        Maximum time for the Fourier transform of the generator voltage
        in seconds. Default is 1e-7 s.
        The maximum time must be large enough to capture the lowest
        frequency component of the generator voltage.

    Notes
    -----
    The transmission line is assumed to be perfect, meaning that there is no
    loss in the line and that the wave velocity is constant.

    For a perfect transmission line, the voltage and current satisfy the wave
    equation (or telegrapher's equation without losses):

    .. math::

        \begin{aligned}
            \frac{\partial^2 V}{\partial t^2}(x, t)
            & = c^2 \frac{\partial^2 V}{\partial x^2}(x, t) \\
            \frac{\partial^2 I}{\partial t^2}(x, t)
            & = c^2 \frac{\partial^2 I}{\partial x^2}(x, t)
        \end{aligned}


    where:

    - :math:`V` is the voltage in Volt,
    - :math:`I` is the current in Ampere,
    - :math:`c` is the wave velocity in m/s,
    - :math:`x` is the position along the transmission line in meter,
    - :math:`t` is the time in second.

    To fully describe the behavior of the transmission line, we need to
    specify the boundary conditions at the ends of the transmission line.
    """

    def __init__(
        self,
        cable: PerfectCable,
        generator: ComplexImpedanceBaseGenerator,
        load: ComplexImpedanceBaseLoad,
        nb_points_ft: int = 1000,
        max_time_ft: float = 1e-7,
    ):
        if not isinstance(cable, PerfectCable):
            raise TypeError("`cable` must be an instance of `PerfectCable`.")
        if not isinstance(generator, ComplexImpedanceBaseGenerator):
            raise TypeError(
                "`generator` must be an instance of"
                " `ComplexImpedanceBaseGenerator`."
            )
        if not isinstance(load, ComplexImpedanceBaseLoad):
            raise TypeError(
                "`load` must be an instance of `ComplexImpedanceBaseLoad`."
            )

        if load.time_varying:
            raise ValueError(
                "The load must be a steady impedance, i.e., not time-varying."
            )

        super().__init__(cable, generator, load)

        # Generator parameters.
        self.Z_g: Callable[[np.ndarray], np.ndarray] = (
            generator.generator_impedance
        )
        self.V_g: Callable[[float], float] = generator.generator_voltage

        self.V_g_hat: np.ndarray
        self.f: np.ndarray
        self.V_g_hat, self.f = self.fourier_transform_generator_voltage(
            nb_points_ft=nb_points_ft, max_time_ft=max_time_ft
        )

        # Load parameters.
        self.Z_l: Callable[[np.ndarray], np.ndarray] = load.load_impedance

        # Pre-compute the frequency-domain coefficients of the total
        # incident/reflected waves. They depend only on frequency (not on
        # time or position), so caching them here avoids recomputing the
        # reflection coefficients and the geometric-series denominator on
        # every `V_incident_total`/`V_reflected_total` call during `solve`.
        self._omega: np.ndarray
        self._incident_coefficient: np.ndarray
        self._reflected_coefficient: np.ndarray
        self._precompute_total_wave_coefficients()

    def _precompute_total_wave_coefficients(self) -> None:
        r"""Cache the frequency-domain total-wave coefficients.

        The total incident and reflected waves are inverse transforms

        .. math::

            V_i(t) = \Re \sum_\omega C_i(\omega)\, e^{j \omega t}, \quad
            V_r(t) = \Re \sum_\omega C_r(\omega)\, e^{j \omega t},

        whose coefficients :math:`C_i`, :math:`C_r` are independent of time
        and position. They are computed once here from the geometric-series
        sum of the multiple reflections.
        """
        omega = 2 * np.pi * self.f
        gamma_g = self.gamma_g(self.f)
        gamma_l = self.gamma_l(self.f)
        alpha_g = self.alpha_g(self.f)
        phase = np.exp(-1j * omega * 2 * self.L / self.c)

        incident_num = alpha_g * self.V_g_hat
        reflected_num = incident_num * gamma_l * phase
        denominator = 1 - gamma_g * gamma_l * phase
        with np.errstate(divide="ignore", invalid="ignore"):
            incident_coefficient = incident_num / denominator
            reflected_coefficient = reflected_num / denominator
        # At the DC point (omega = 0) a purely reflective generator and load
        # (Gamma_g = Gamma_l = 1, e.g. a capacitive generator feeding a
        # capacitive load) make the denominator vanish; there the numerator
        # vanishes too (alpha_g = 0, the series capacitor blocks DC), so the
        # 0/0 term launches no wave and is set to 0.
        self._omega = omega
        self._incident_coefficient = np.where(
            (denominator == 0) & (incident_num == 0),
            0.0,
            incident_coefficient,
        )
        self._reflected_coefficient = np.where(
            (denominator == 0) & (reflected_num == 0),
            0.0,
            reflected_coefficient,
        )

    def fourier_transform_generator_voltage(
        self,
        nb_points_ft: int,
        max_time_ft: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fourier transform of the generator voltage.

        Parameters
        ----------
        nb_points_ft : int
            Number of points for the Fourier transform.
        max_time_ft : float
            Maximum time for the Fourier transform in seconds.

        Returns
        -------
        tuple of numpy.ndarray, numpy.ndarray
            Tuple of the Fourier transform of the generator voltage and
            the frequencies.

        Notes
        -----
        The Fourier transform is computed using the Fast Fourier Transform
        (FFT) algorithm. The time vector is defined from 0 to `max_time_ft`
        with `nb_points_ft` points. The frequency vector is computed using
        the FFT frequency function.

        The Shannon-Nyquist criterion is checked to ensure that the time
        step is small enough to capture the highest frequency component
        of the generator voltage. If the time step is larger than 1 ns,
        a warning is issued.

        In NRP Discharge, the maximum frequency encountered is around
        the inverse of the rise time of the generator voltage.
        For a rise time of 2 ns, the maximum frequency is around 500 MHz.
        According to the Shannon-Nyquist criterion, the sampling
        frequency must be at least twice the maximum frequency, so at
        least 1 GHz. This corresponds to a time step of 1 ns.
        """
        # Define the time vector for the Fourier transform.
        times = np.linspace(0, max_time_ft, nb_points_ft, endpoint=False)

        dt = times[1] - times[0]  # time step

        # Shannon-Nyquist criterion.
        if dt > 1e-9:
            warn(
                f"Warning: dt = {dt:.2e} s seems to be "
                "too large for Fourier Transform of NRP Discharge."
            )

        # Generator voltage in time domain.
        generator_voltage = np.array([self.V_g(t) for t in times])
        # Generator voltage in frequency domain.
        generator_voltage_hat = np.fft.fft(generator_voltage, norm="forward")
        # Frequency.
        frequencies = np.fft.fftfreq(nb_points_ft, dt)

        return generator_voltage_hat, frequencies

    def gamma_g(self, frequency: np.ndarray) -> np.ndarray:
        r"""Calculate the generator reflection coefficient.

        Parameters
        ----------
        frequency : numpy.ndarray
            Frequency array in Hz.

        Returns
        -------
        numpy.ndarray
            Generator reflection coefficient, dimensionless.

        Notes
        -----
        The generator reflection coefficient is defined as:

        .. math::

            \Gamma_g(f) = \frac{Z_g(f) - Z_c}{Z_g(f) + Z_c}

        where:

        - :math:`Z_g(f)` is the generator impedance in Ohm,
        - :math:`f` is the frequency in Hz,
        - :math:`Z_c` is the characteristic impedance of the transmission line
          in Ohm.

        At a frequency where the generator impedance is infinite (open
        circuit, e.g. the DC component of a purely capacitive generator),
        the reflection coefficient tends to its open-circuit limit
        :math:`\Gamma_g \to 1`.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            Z_g = self.Z_g(frequency)
            gamma_g = (Z_g - self.Z_c) / (Z_g + self.Z_c)
        # Open-circuit limit: |Z_g| -> infinity gives gamma_g -> 1.
        return np.where(np.isfinite(gamma_g), gamma_g, 1.0)

    def gamma_l(self, frequency: np.ndarray) -> np.ndarray:
        r"""Calculate the load reflection coefficient.

        Parameters
        ----------
        frequency : numpy.ndarray
            Frequency array in Hz.

        Returns
        -------
        numpy.ndarray
            Load reflection coefficient, dimensionless.

        Notes
        -----
        The load reflection coefficient is defined as:

        .. math::

            \Gamma_l(f) = \frac{Z_l(f) - Z_c}{Z_l(f) + Z_c}

        where:

        - :math:`Z_l(f)` is the load impedance in Ohm,
        - :math:`f` is the frequency in Hz,
        - :math:`Z_c` is the characteristic impedance of the transmission line
          in Ohm.

        At a frequency where the load impedance is infinite (open circuit,
        e.g. the DC component of a purely capacitive load), the reflection
        coefficient tends to its open-circuit limit :math:`\Gamma_l \to 1`.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            Z_l = self.Z_l(frequency)
            gamma_l = (Z_l - self.Z_c) / (Z_l + self.Z_c)
        # Open-circuit limit: |Z_l| -> infinity gives gamma_l -> 1.
        return np.where(np.isfinite(gamma_l), gamma_l, 1.0)

    def alpha_g(self, frequency: np.ndarray) -> np.ndarray:
        r"""Calculate the generator resistance voltage divider.

        Parameters
        ----------
        frequency : numpy.ndarray
            Frequency array in Hz.

        Returns
        -------
        numpy.ndarray
            Generator resistance voltage divider, dimensionless.

        Notes
        -----
        The generator resistance voltage divider is defined as:

        .. math::

            \alpha_g(f) = \frac{Z_c}{Z_g(f) + Z_c}

        where:

        - :math:`Z_g(f)` is the generator impedance in Ohm,
        - :math:`f` is the frequency in Hz,
        - :math:`Z_c` is the characteristic impedance of the transmission line
          in Ohm.

        At a frequency where the generator impedance is infinite (open
        circuit, e.g. the DC component of a purely capacitive generator), no
        voltage is launched into the line, so :math:`\alpha_g \to 0`.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            Z_g = self.Z_g(frequency)
            alpha_g = 1 / (1 + Z_g / self.Z_c)
        # Open-circuit limit: |Z_g| -> infinity gives alpha_g -> 0.
        return np.where(np.isfinite(alpha_g), alpha_g, 0.0)

    def V_incident(self, t: float, n: int) -> float:
        r"""Incident wave.

        Compute the incident wave of generation :math:`n` at time :math:`t`.

        Parameters
        ----------
        t : float
            Time in seconds.
        n : int
            Generation number.

        Returns
        -------
        float
            Incident wave value in volts.

        Notes
        -----
        The incident wave of generation :math:`n` is defined as:

        .. math::

            V_i^n(t) =
            \Re \left\{
                \int_{-\infty}^{+\infty}
                    \alpha_g(\omega)
                    \hat{V_g}(\omega)
                    e^{j \omega \left(t - \frac{2 n L}{c}\right)}
                    \left(\Gamma_g(\omega) \Gamma_l(\omega)\right)^n
            \right\}
        """
        # Get the reflection coefficient at the generator.
        gamma_g_ω = self.gamma_g(self.f)

        # Get the reflection coefficient at the load.
        gamma_l_ω = self.gamma_l(self.f)

        # Get the attenuation factor at the generator.
        alpha_g_ω = self.alpha_g(self.f)

        # Compute the incident wave.
        v_incident = (
            alpha_g_ω
            * self.V_g_hat
            * np.exp(2j * np.pi * self.f * (t - 2 * n * self.L / self.c))
            * (gamma_g_ω * gamma_l_ω) ** n
        )

        return np.real(np.sum(v_incident))

    def V_reflected(self, t: float, n: int) -> float:
        r"""Reflected wave.

        Compute the reflected wave of generation :math:`n` at time :math:`t`.

        Parameters
        ----------
        t : float
            Time in seconds.
        n : int
            Generation number.

        Returns
        -------
        float
            Incident wave value in volts.

        Notes
        -----
        The reflected wave of generation :math:`n` is defined as:

        .. math::

            V_r^n(t) =
            \Re \left\{
                \int_{-\infty}^{+\infty}
                    \gamma_l(\omega)
                    \alpha_g(\omega)
                    \hat{V_g}(\omega)
                    e^{j \omega \left(t - \frac{2 (n + 1) L}{c}\right)}
                    \left(\Gamma_g(\omega) \Gamma_l(\omega)\right)^n
            \right\}
        """
        # Get the reflection coefficient at the generator.
        gamma_g_ω = self.gamma_g(self.f)

        # Get the reflection coefficient at the load.
        gamma_l_ω = self.gamma_l(self.f)

        # Get the attenuation factor at the generator.
        alpha_g_ω = self.alpha_g(self.f)

        # Compute the reflected wave.
        v_reflected = (
            gamma_l_ω
            * alpha_g_ω
            * self.V_g_hat
            * np.exp(2j * np.pi * self.f * (t - 2 * (n + 1) * self.L / self.c))
            * (gamma_g_ω * gamma_l_ω) ** n
        )

        return np.real(np.sum(v_reflected))

    def V_incident_total(self, t: float) -> float:
        # Inverse transform of the cached incident-wave coefficients.
        return np.real(
            np.sum(self._incident_coefficient * np.exp(1j * self._omega * t))
        )

    def V_reflected_total(self, t: float) -> float:
        # Inverse transform of the cached reflected-wave coefficients.
        return np.real(
            np.sum(self._reflected_coefficient * np.exp(1j * self._omega * t))
        )
