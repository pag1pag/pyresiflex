import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from pyresiflex.load.time_varying_resistance import (
    ConstantResistance,
    PlasmaResistanceLinearFall,
)
from pyresiflex.solver.base_solution import BaseSolution
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution


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


@pytest.fixture
def solution(perfect_cable, trapezoidal_generator):
    """Return an unsolved purely resistive solution (constant load)."""
    return PurelyResistiveSolution(
        cable=perfect_cable,
        generator=trapezoidal_generator,
        load=ConstantResistance(R=200.0),
    )


@pytest.fixture
def solved_solution(perfect_cable, trapezoidal_generator):
    """Return a solution already solved on a position/time grid."""
    load = PlasmaResistanceLinearFall(
        Z_start=1e3, Z_end=10.0, t_start_fall=10e-9, t_end_fall=20e-9
    )
    sol = PurelyResistiveSolution(
        cable=perfect_cable, generator=trapezoidal_generator, load=load
    )
    sol.solve(np.linspace(0, perfect_cable.L, 6), np.linspace(0, 25e-9, 5))
    return sol


def test_solve_grid_matches_pointwise_V_and_I(solution):
    """`solve` on a grid is consistent with pointwise `V` and `I`."""
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


def test_solve_single_position_flattens(solution):
    """A scalar position produces 1-D output arrays."""
    t = np.linspace(0, 20e-9, 7)
    solution.solve(0.1, t)
    assert solution.voltage.shape == (7,)
    assert solution.current.shape == (7,)
    assert solution.x.shape == (1,)


def test_solve_scalar_time(solution):
    """A scalar time is accepted as well."""
    solution.solve(0.1, 5e-9)
    assert solution.voltage.shape == (1,)


@pytest.mark.parametrize(
    "x, t, match",
    [
        (np.zeros((2, 2)), np.linspace(0, 1e-9, 3), "x must be a 1D array"),
        ("bad", np.linspace(0, 1e-9, 3), "x must be a 1D array"),
        (np.array([-1.0]), np.linspace(0, 1e-9, 3), "between 0 and L"),
        (np.array([10.0]), np.linspace(0, 1e-9, 3), "between 0 and L"),
        (0.1, np.zeros((2, 2)), "t must be a 1D array"),
        (0.1, "bad", "t must be a 1D array"),
    ],
)
def test_solve_invalid_inputs(solution, x, t, match):
    with pytest.raises(ValueError, match=match):
        solution.solve(x, t)


def test_base_solution_type_checks(perfect_cable, trapezoidal_generator):
    """The base constructor validates cable, generator and load types."""
    cable, gen = perfect_cable, trapezoidal_generator
    load = ConstantResistance(R=200.0)
    with pytest.raises(TypeError, match=r"`cable` must be an instance"):
        _ConcreteSolution("bad", gen, load)  # type: ignore
    with pytest.raises(TypeError, match=r"`generator` must be an instance"):
        _ConcreteSolution(cable, "bad", load)  # type: ignore
    with pytest.raises(TypeError, match=r"`load` must be an instance"):
        _ConcreteSolution(cable, gen, "bad")  # type: ignore


def test_purely_resistive_solution_type_checks(
    perfect_cable, trapezoidal_generator
):
    """`PurelyResistiveSolution` validates cable, generator and load types."""
    cable, gen = perfect_cable, trapezoidal_generator
    load = ConstantResistance(R=200.0)
    with pytest.raises(TypeError, match=r"`cable` must be an instance"):
        PurelyResistiveSolution("bad", gen, load)  # type: ignore
    with pytest.raises(TypeError, match=r"`generator` must be an instance"):
        PurelyResistiveSolution(cable, "bad", load)  # type: ignore
    with pytest.raises(TypeError, match=r"`load` must be an instance"):
        PurelyResistiveSolution(cable, gen, "bad")  # type: ignore


def test_abstract_wave_methods_raise(perfect_cable, trapezoidal_generator):
    """The abstract incident/reflected waves raise NotImplementedError."""
    load = ConstantResistance(R=200.0)
    sol = _SuperCallSolution(perfect_cable, trapezoidal_generator, load)
    with pytest.raises(NotImplementedError, match="incident wave"):
        sol.V_incident(0.0, 0)
    with pytest.raises(NotImplementedError, match="reflected wave"):
        sol.V_reflected(0.0, 0)


def _run_frames(ani: FuncAnimation, n_frames: int) -> None:
    """Invoke the animation update function for every frame."""
    # `_func` is the (private) frame-update closure built in `animation`.
    update = getattr(ani, "_func")
    for idx in range(n_frames):
        update(idx)


def test_animation_with_current(solved_solution):
    ani = solved_solution.animation(with_current=True, show=False)
    assert isinstance(ani, FuncAnimation)
    _run_frames(ani, len(solved_solution.t))


def test_animation_without_current(solved_solution):
    ani = solved_solution.animation(with_current=False, show=False)
    assert isinstance(ani, FuncAnimation)
    _run_frames(ani, len(solved_solution.t))


def test_animation_explicit_limits(solved_solution):
    ani = solved_solution.animation(
        with_current=True,
        show=False,
        y_min_max_voltage=(-10.0, 10.0),
        y_min_max_current=(-5.0, 5.0),
    )
    assert isinstance(ani, FuncAnimation)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"y_min_max_voltage": [0.0, 1.0]}, r"`y_min_max_voltage` must be"),
        ({"y_min_max_current": [0.0, 1.0]}, r"`y_min_max_current` must be"),
    ],
)
def test_animation_invalid_limits(solved_solution, kwargs, match):
    with pytest.raises(ValueError, match=match):
        solved_solution.animation(with_current=True, show=False, **kwargs)


def test_animation_show(solved_solution, monkeypatch):
    called = {}
    monkeypatch.setattr(
        "matplotlib.pyplot.show", lambda *a, **k: called.setdefault("s", True)
    )
    solved_solution.animation(with_current=True, show=True)
    assert called.get("s", False)


if __name__ == "__main__":
    pytest.main([__file__])
