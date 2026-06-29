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
def solution(
    perfect_cable: PerfectCable,
    trapezoidal_generator: TrapezoidalGenerator,
) -> PurelyResistiveSolution:
    """Return an unsolved purely resistive solution (constant load)."""
    return PurelyResistiveSolution(
        cable=perfect_cable,
        generator=trapezoidal_generator,
        load=ConstantResistance(R=200.0),
    )


@pytest.fixture
def solved_solution(
    perfect_cable: PerfectCable,
    trapezoidal_generator: TrapezoidalGenerator,
) -> PurelyResistiveSolution:
    """Return a solution already solved on a position/time grid."""
    load = PlasmaResistanceLinearFall(
        Z_start=1e3, Z_end=10.0, t_start_fall=10e-9, t_end_fall=20e-9
    )
    sol = PurelyResistiveSolution(
        cable=perfect_cable, generator=trapezoidal_generator, load=load
    )
    sol.solve(np.linspace(0, perfect_cable.L, 6), np.linspace(0, 25e-9, 5))
    return sol


def test_solve_grid_matches_pointwise_V_and_I(
    solution: PurelyResistiveSolution,
) -> None:
    """Check `solve` on a grid matches pointwise `V` and `I`.

    The grid arrays stored by ``solve`` must equal element-by-element the
    values returned by the scalar ``V``/``I`` accessors, and the derived
    ``power``/``energy`` arrays must follow from them.
    """
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


def test_solve_single_position_flattens(
    solution: PurelyResistiveSolution,
) -> None:
    """Verify a scalar position yields 1-D output arrays.

    A single position collapses the spatial axis, so the voltage and
    current arrays are 1-D over time while ``x`` keeps a single entry.
    """
    t = np.linspace(0, 20e-9, 7)
    solution.solve(0.1, t)
    assert solution.voltage.shape == (7,)
    assert solution.current.shape == (7,)
    assert solution.x.shape == (1,)


def test_solve_scalar_time(solution: PurelyResistiveSolution) -> None:
    """Ensure a scalar time is accepted by `solve`.

    A single position and a single time produce a length-one voltage array.
    """
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
def test_solve_invalid_inputs(
    solution: PurelyResistiveSolution, x, t, match: str
) -> None:
    """Check `solve` rejects wrong-shape or out-of-range positions/times.

    ``x`` and ``t`` are deliberately invalid mixed types (bad shapes,
    strings, out-of-range values) fed to ``solve`` to trigger errors, so
    they are left unannotated on purpose.
    """
    with pytest.raises(ValueError, match=match):
        solution.solve(x, t)


def test_base_solution_type_checks(
    perfect_cable: PerfectCable,
    trapezoidal_generator: TrapezoidalGenerator,
) -> None:
    """Check the base constructor validates cable/generator/load types.

    Passing a wrong type for each argument in turn must raise ``TypeError``
    with a message identifying the offending argument.
    """
    cable, gen = perfect_cable, trapezoidal_generator
    load = ConstantResistance(R=200.0)
    with pytest.raises(TypeError, match=r"`cable` must be an instance"):
        _ConcreteSolution("bad", gen, load)  # type: ignore
    with pytest.raises(TypeError, match=r"`generator` must be an instance"):
        _ConcreteSolution(cable, "bad", load)  # type: ignore
    with pytest.raises(TypeError, match=r"`load` must be an instance"):
        _ConcreteSolution(cable, gen, "bad")  # type: ignore


def test_purely_resistive_solution_type_checks(
    perfect_cable: PerfectCable,
    trapezoidal_generator: TrapezoidalGenerator,
) -> None:
    """Check `PurelyResistiveSolution` validates constructor types.

    Like the base class, a wrong cable/generator/load type must raise
    ``TypeError`` naming the offending argument.
    """
    cable, gen = perfect_cable, trapezoidal_generator
    load = ConstantResistance(R=200.0)
    with pytest.raises(TypeError, match=r"`cable` must be an instance"):
        PurelyResistiveSolution("bad", gen, load)  # type: ignore
    with pytest.raises(TypeError, match=r"`generator` must be an instance"):
        PurelyResistiveSolution(cable, "bad", load)  # type: ignore
    with pytest.raises(TypeError, match=r"`load` must be an instance"):
        PurelyResistiveSolution(cable, gen, "bad")  # type: ignore


def test_abstract_wave_methods_raise(
    perfect_cable: PerfectCable,
    trapezoidal_generator: TrapezoidalGenerator,
) -> None:
    """Check the abstract wave methods raise ``NotImplementedError``.

    ``_SuperCallSolution`` delegates to the base implementations, so calling
    the incident/reflected waves must raise ``NotImplementedError``.
    """
    load = ConstantResistance(R=200.0)
    sol = _SuperCallSolution(perfect_cable, trapezoidal_generator, load)
    with pytest.raises(NotImplementedError, match="incident wave"):
        sol.V_incident(0.0, 0)
    with pytest.raises(NotImplementedError, match="reflected wave"):
        sol.V_reflected(0.0, 0)


def _render_all_frames(ani: FuncAnimation) -> None:
    """Render every frame of ``ani`` to exercise the update closure.

    ``to_jshtml`` drives the frame-update closure built inside ``animation``
    for all frames through matplotlib's public API (no reliance on private
    attributes), so the drawing code is exercised without an event loop.
    """
    assert ani.to_jshtml()


def test_animation_with_current(
    solved_solution: PurelyResistiveSolution,
) -> None:
    """Check `animation` with current returns a runnable ``FuncAnimation``.

    Render every frame to confirm the voltage-and-current update closure
    runs without error.
    """
    ani = solved_solution.animation(with_current=True, show=False)
    assert isinstance(ani, FuncAnimation)
    _render_all_frames(ani)


def test_animation_without_current(
    solved_solution: PurelyResistiveSolution,
) -> None:
    """Check `animation` without current returns a runnable animation.

    Render every frame to confirm the voltage-only update closure runs.
    """
    ani = solved_solution.animation(with_current=False, show=False)
    assert isinstance(ani, FuncAnimation)
    _render_all_frames(ani)


def test_animation_explicit_limits(
    solved_solution: PurelyResistiveSolution,
) -> None:
    """Check `animation` accepts explicit voltage/current axis limits.

    Supplying both ``y_min_max_*`` tuples must still build a valid
    ``FuncAnimation``.
    """
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
def test_animation_invalid_limits(
    solved_solution: PurelyResistiveSolution,
    kwargs,  # bare: spread via **kwargs into typed params (keeps ty happy)
    match: str,
) -> None:
    """Check `animation` rejects malformed axis-limit arguments.

    A list (instead of the expected 2-tuple) for either limit must raise
    ``ValueError`` naming the offending keyword.
    """
    with pytest.raises(ValueError, match=match):
        solved_solution.animation(with_current=True, show=False, **kwargs)


def test_animation_show(
    solved_solution: PurelyResistiveSolution,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check `animation` calls ``pyplot.show`` when ``show=True``.

    Patch ``matplotlib.pyplot.show`` so the test records whether it ran
    rather than opening a window.
    """
    called = {}
    monkeypatch.setattr(
        "matplotlib.pyplot.show", lambda *a, **k: called.setdefault("s", True)
    )
    solved_solution.animation(with_current=True, show=True)
    assert called.get("s", False)


if __name__ == "__main__":
    pytest.main([__file__])
