class PerfectCable:
    r"""Perfect transmission line.

    The transmission line is defined by its length, impedance
    and wave velocity.

    The transmission line is assumed to be perfect, meaning that there is no
    loss in the line and that the wave velocity is constant.

    Parameters
    ----------
    L : float
        Length of the transmission line in meters.
    Z_c : float
        Characteristic impedance of the transmission line in Ohm.
    c : float
        Wave velocity in the transmission line in m/s.

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

    For more details, see e.g. [Rao2009]_ or [Inan2016]_.

    To fully describe the behavior of the transmission line, we need to
    specify the boundary conditions at the ends of the transmission line.

    Examples
    --------
    >>> from pyresiflex.cable.cable import PerfectCable
    >>> cable = PerfectCable(L=10, Z_c=50, c=2e8)
    >>> cable.L
    10
    >>> cable.Z_c
    50
    >>> cable.c
    200000000.0

    .. minigallery:: pyresiflex.cable.cable.PerfectCable
    """

    def __init__(
        self,
        L: float,
        Z_c: float,
        c: float,
    ):
        self.check_parameters(L, Z_c, c)
        self.L: float = L
        self.Z_c: float = Z_c
        self.c: float = c

    @staticmethod
    def check_parameters(L: float, Z_c: float, c: float):
        """Check the parameters of the transmission line."""
        # .. Type checking.
        if not isinstance(L, (int, float)):
            raise TypeError(
                "Length of the transmission line `L` must be a number."
            )
        if not isinstance(Z_c, (int, float)):
            raise TypeError("Characteristic impedance `Z_c` must be a number.")
        if not isinstance(c, (int, float)):
            raise TypeError("Wave velocity `c` must be a number.")
        # .. Value checking.
        if L <= 0:
            raise ValueError(
                "Length of the transmission line `L` must be positive."
            )
        if Z_c <= 0:
            raise ValueError(
                "Characteristic impedance `Z_c` must be positive."
            )
        if c <= 0:
            raise ValueError("Wave velocity `c` must be positive.")
        if c > 3e8:
            raise ValueError(
                "Wave velocity `c` cannot exceed the speed of light."
            )
