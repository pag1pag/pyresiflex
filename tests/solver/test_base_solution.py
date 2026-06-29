import matplotlib
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.generator_real_impedance import TrapezoidalGenerator
from pyresiflex.load.time_varying_resistance import (
    ConstantResistance,
    PlasmaResistanceLinearFall,
)
from pyresiflex.solver.base_solution import BaseSolution
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

matplotlib.use("Agg")  # Non-interactive backend for testing.


class _ConcreteSolution(BaseSolution):
    """Minimal concrete solution returning zero waves."""

    def V_incident(self, t: float, n: int) -> float:
        return 0.0

    def V_reflected(self, t: float, n: int) -> float:
        return 0.0


class _SuperCallSolution(BaseSolution):
    """Concrete solution delegating to the abstract base implementations."""

    def V_incident(self, t: float, n: int) -> float:
        return super().V_incident(t, n)

    def V_reflected(self, t: float, n: int) -> float:
        return super().V_reflected(t, n)


def _make_solution() -> PurelyResistiveSolution:
    cable = PerfectCable(L=0.3, Z_c=75.0, c=1.66e8)
    generator = TrapezoidalGenerator(R_g=1.0, U_on=10e3)
    load = ConstantResistance(R=200.0)
    return PurelyResistiveSolution(cable=cable, generator=generator, load=load)


def test_solve_grid_matches_pointwise_V_and_I():
    """`solve` on a grid is consistent with pointwise `V` and `I`."""
    solution = _make_solution()
    x = np.linspace(0, solution.L, 5)
    t = np.linspace(0, 20e-9, 7)
    solution.solve(x, t)

    voltage_grid = np.array([[solution.V(xi, tj) for tj in t] for xi in x])
    current_grid = np.array([[solution.I(xi, tj) for tj in t] for xi in x])

    assert np.allclose(solution.voltage, voltage_grid)
    assert np.allclose(solution.current, current_grid)
    # Power is the product of voltage and current.
    assert np.allclose(solution.power, solution.voltage * solution.current)
    # Cumulative energy starts at zero at the first time.
    assert np.allclose(solution.energy[:, 0], 0.0)
    assert solution.voltage.shape == (5, 7)


def test_solve_single_position_flattens():
    """A scalar position produces 1-D output arrays."""
    solution = _make_solution()
    t = np.linspace(0, 20e-9, 7)
    solution.solve(0.1, t)
    assert solution.voltage.shape == (7,)
    assert solution.current.shape == (7,)
    assert solution.x.shape == (1,)


def test_solve_scalar_time():
    """A scalar time is accepted as well."""
    solution = _make_solution()
    solution.solve(0.1, 5e-9)
    assert solution.voltage.shape == (1,)


@pytest.mark.parametrize(
    "x, t",
    [
        (np.zeros((2, 2)), np.linspace(0, 1e-9, 3)),  # 2-D x
        ("bad", np.linspace(0, 1e-9, 3)),  # wrong x type
        (np.array([-1.0]), np.linspace(0, 1e-9, 3)),  # x < 0
        (np.array([10.0]), np.linspace(0, 1e-9, 3)),  # x > L
        (0.1, np.zeros((2, 2))),  # 2-D t
        (0.1, "bad"),  # wrong t type
    ],
)
def test_solve_invalid_inputs(x, t):
    solution = _make_solution()
    with pytest.raises(ValueError):
        solution.solve(x, t)


def test_base_solution_type_checks():
    """The base constructor validates cable, generator and load types."""
    cable = PerfectCable(L=0.3, Z_c=75.0, c=1.66e8)
    generator = TrapezoidalGenerator(R_g=1.0)
    load = ConstantResistance(R=200.0)

    with pytest.raises(TypeError):
        _ConcreteSolution("bad", generator, load)  # type: ignore
    with pytest.raises(TypeError):
        _ConcreteSolution(cable, "bad", load)  # type: ignore
    with pytest.raises(TypeError):
        _ConcreteSolution(cable, generator, "bad")  # type: ignore


def test_purely_resistive_solution_type_checks():
    """`PurelyResistiveSolution` validates cable, generator and load types."""
    cable = PerfectCable(L=0.3, Z_c=75.0, c=1.66e8)
    generator = TrapezoidalGenerator(R_g=1.0)
    load = ConstantResistance(R=200.0)

    with pytest.raises(TypeError):
        PurelyResistiveSolution("bad", generator, load)  # type: ignore
    with pytest.raises(TypeError):
        PurelyResistiveSolution(cable, "bad", load)  # type: ignore
    with pytest.raises(TypeError):
        PurelyResistiveSolution(cable, generator, "bad")  # type: ignore


def test_abstract_wave_methods_raise():
    """The abstract incident/reflected waves raise NotImplementedError."""
    cable = PerfectCable(L=0.3, Z_c=75.0, c=1.66e8)
    generator = TrapezoidalGenerator(R_g=1.0)
    load = ConstantResistance(R=200.0)
    solution = _SuperCallSolution(cable, generator, load)
    with pytest.raises(NotImplementedError):
        solution.V_incident(0.0, 0)
    with pytest.raises(NotImplementedError):
        solution.V_reflected(0.0, 0)


def _make_solved_solution() -> PurelyResistiveSolution:
    cable = PerfectCable(L=0.3, Z_c=75.0, c=1.66e8)
    generator = TrapezoidalGenerator(R_g=1.0, U_on=10e3)
    load = PlasmaResistanceLinearFall(
        Z_start=1e3,
        Z_end=10.0,
        t_start_fall=10e-9,
        t_end_fall=20e-9,
    )
    solution = PurelyResistiveSolution(
        cable=cable, generator=generator, load=load
    )
    x = np.linspace(0, cable.L, 6)
    t = np.linspace(0, 25e-9, 5)
    solution.solve(x, t)
    return solution


def _run_frames(ani: FuncAnimation, n_frames: int) -> None:
    """Invoke the animation update function for every frame."""
    # `_func` is the (private) frame-update closure built in `animation`.
    update = getattr(ani, "_func")
    for idx in range(n_frames):
        update(idx)


def test_animation_with_current():
    solution = _make_solved_solution()
    ani = solution.animation(with_current=True, show=False)
    assert isinstance(ani, FuncAnimation)
    _run_frames(ani, len(solution.t))


def test_animation_without_current():
    solution = _make_solved_solution()
    ani = solution.animation(with_current=False, show=False)
    assert isinstance(ani, FuncAnimation)
    _run_frames(ani, len(solution.t))


def test_animation_explicit_limits():
    solution = _make_solved_solution()
    ani = solution.animation(
        with_current=True,
        show=False,
        y_min_max_voltage=(-10.0, 10.0),
        y_min_max_current=(-5.0, 5.0),
    )
    assert isinstance(ani, FuncAnimation)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"y_min_max_voltage": [0.0, 1.0]},  # list, not a tuple
        {"y_min_max_current": [0.0, 1.0]},  # list, not a tuple
    ],
)
def test_animation_invalid_limits(kwargs):
    solution = _make_solved_solution()
    with pytest.raises(ValueError):
        solution.animation(with_current=True, show=False, **kwargs)


def test_animation_show(monkeypatch):
    solution = _make_solved_solution()
    called = {}
    monkeypatch.setattr(
        "matplotlib.pyplot.show", lambda *a, **k: called.setdefault("s", True)
    )
    solution.animation(with_current=True, show=True)
    assert called.get("s", False)


if __name__ == "__main__":
    pytest.main([__file__])
