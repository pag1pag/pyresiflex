# [PyResiFlex](pag1pag.github.io/pyresiflex/)

[![Build Status](https://github.com/pag1pag/pyresiflex/actions/workflows/on-push.yml/badge.svg)](https://github.com/pag1pag/pyresiflex/actions/workflows/on-push.yml/badge.svg)
[![Coverage Status](https://raw.githubusercontent.com/pag1pag/pyresiflex/refs/heads/coverage-badge/coverage.svg?raw=true)](https://raw.githubusercontent.com/pag1pag/pyresiflex/refs/heads/coverage-badge/coverage.svg?raw=true)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=958704955&skip_quickstart=true)

**PyResiFlex** is a simple set of tools in **Py**thon to obtain load (plasma) **Resi**stance from analysis of pulse re**Flex**ions.

Full user documentation (advanced install and examples) are available on the [PyResiFlex Website](pag1pag.github.io/pyresiflex/).

## Getting Started

### Install

Assuming you have Python installed with the [Anaconda](https://www.anaconda.com/download/) distribution you can use `pip`:

```bash
pip install radis
```

### Quick Start

Calculate a CO equilibrium spectrum from the HITRAN database:

```python
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

from pyresiflex.cable.cable import PerfectCable
from pyresiflex.generator.generator_real_impedance import ConstantGenerator
from pyresiflex.load.time_varying_resistance import PlasmaResistanceLinearFall
from pyresiflex.solver.purely_resistive_solution import PurelyResistiveSolution

# Create a purely resistive solution with a time-varying load resistance.
solution = PurelyResistiveSolution(
    cable=PerfectCable(L=5, Z_c=75, c=2e8),
    generator=ConstantGenerator(R_g=1, U_g=5e3),
    load=PlasmaResistanceLinearFall(
        Z_start=1e2, Z_end=10, t_start_fall=20e-9, t_end_fall=30e-9
    ),
)

# Solve the system at specific time points.
times = np.linspace(0, 40e-9, 1000)
# Here, the solution is computed at 6 meters.
solution.solve(x=5, t=times)

# Plot the voltage response over time.
fig, ax = plt.subplots()
ax.plot(times * 1e9, solution.voltage * 1e-3, color="k")
ax.set_xlabel(r"$\mathregular{t \, [ns]}$")
ax.set_ylabel(r"$\mathregular{V \, [kV]}$")
ax.set_title("Load voltage against time")
plt.show()
```

## Example

![GIF showing the reproduction of the Minesi2022 experiment](./docs/img/reproduce_Minesi2022_experiments.gif)

[Example](https://pag1pag.github.io/pyresiflex/auto_examples/article_figures/plot_reproduce_Minesi2022_experiment.html) of the reproduction of the [Minesi2022](https://doi.org/10.1088/1361-6595/ac5cd4) experiment using PyResiFlex.

## Workflow for developers/contributors

GitHub Codespaces has been set up, so you can click on this button to quickly code and run the examples: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=958704955&skip_quickstart=true)

You can also follow the instructions below, to set up the project on your machine:

- First, clone the repository with `git clone https://github.com/pag1pag/pyresiflex.git`.
- Since this package uses [uv](https://docs.astral.sh/uv/), install it by following [instructions on their website](https://docs.astral.sh/uv/getting-started/installation/).
- Update this Python package manager with `uv self update`.
- Run `uv sync` to create a virtual environment at `.venv`, with the latest version of Python and all the necessary dependencies.
- To test if the package is working, run `uv run pytest`. All tests should pass.

Next, you just need to activate the virtual environment with:

- (macOS and Linux) `source .venv/bin/activate`
- (Windows) `.venv\Scripts\activate`

Do not forget to activate it each time you are using this project.

Before pushing to GitHub, run the following commands in a terminal (with the virtual environment activated):

1. Update dependencies with `just update-env`.
1. Run quality assurance checks (code linting) with `just qa`.
1. Run type checks with `just type-check`.
1. Run unit tests with `just tests`.
1. Run unit tests with coverage and generate a badge with `just tests-cov`.
1. Build the documentation with `just build-docs`.

You could also run `just` to run all the above commands in one go.

### Manually building the package

If you want to install `pyresiflex` to another environment, you can build the package and install it with `pip`:

- `(pyresiflex) uv build` will build a wheel `/whl` in the folder `./dist`.
- Activating another env, called `other_env`.
- `(other_env) pip install path/to/file.whl` should install `pyresiflex` in `other_env`.

## Who do I talk to?

- Pierre-Antoine Goutier, Spark Cleantech & EM2C Lab, 2024-present, <pierre-antoine.goutier@spark-cleantech.eu>

## References

A list of references used is available at [the reference section](https://pag1pag.github.io/pyresiflex/bibliography.html).

## Note

According to [Oxford Learner's Dictionaries](https://www.oxfordlearnersdictionaries.com/definition/english/reflexion), *reflexion* is an old spelling of *reflection*.
