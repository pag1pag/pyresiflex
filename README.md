# README

[![Build Status](https://github.com/pag1pag/pyresiflex/actions/workflows/on-push.yml/badge.svg)](https://github.com/pag1pag/pyresiflex/actions/workflows/on-push.yml/badge.svg)
[![Coverage Status](https://raw.githubusercontent.com/pag1pag/pyresiflex/refs/heads/coverage-badge/coverage.svg?raw=true)](https://raw.githubusercontent.com/pag1pag/pyresiflex/refs/heads/coverage-badge/coverage.svg?raw=true)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=958704955&skip_quickstart=true)

**PyResiFlex** is a simple set of tools in **Py**thon to obtain load (plasma) **Resi**stance from analysis of pulse re**Flex**ions.

## Quick start

- Simply install the package with `pip install pyresiflex`.
- You should now be able to run the example scripts in the `examples` directory.

## Example

![GIF showing the reproduction of the Minesi2022 experiment](./docs/img/reproduce_Minesi2022_experiments.gif)

Exemple of the reproduction of the [Minesi2022](https://doi.org/10.1088/1361-6595/ac5cd4) experiment using PyResiFlex.

## Documentation

A full set of documentation is available online at
[https://pag1pag.github.io/pyresiflex/](https://pag1pag.github.io/pyresiflex/).

## Workflow for developers/contributors

GitHub codespace has been set up, so you click on this button to quickly code and run the examples: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=958704955&skip_quickstart=true)

You can also follow the instructions below, to set up the project on your machine:

- First, clone the repository. You'll get the package and some test drivers.
- Since this package uses [uv](https://docs.astral.sh/uv/), install it by following [instructions on their website](https://docs.astral.sh/uv/getting-started/installation/).
- Run `uv sync --python 3.13` to create a virtual environment at `.venv`, with `python 3.13` and all the necessary dependencies.
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

## Who do I talk to?

- Pierre-Antoine Goutier, Spark Cleantech & EM2C Lab, 2024-present, <pierre-antoine.goutier@spark-cleantech.eu>

## References

- A list of references are available at [the reference section](https://pag1pag.github.io/pyresiflex/bibliography.html).
- To add a reference, add a new entry to the `./docs/bibliography.rst` file.

## Note

According to the [Oxford's dictionary](https://www.oxfordlearnersdictionaries.com/definition/english/reflexion), *reflexion* is an old spelling of *reflection*.
