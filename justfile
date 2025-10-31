# This file is used by the `just` command runner, available at https://just.systems/man/en/.
# It is similar to a Makefile, but with a simpler syntax and more features.
#
# Run `just --list` to see all the available recipes.
# Run `just <recipe>` to run a specific recipe.

# Configure the shell to use for running commands.
# See https://just.systems/man/en/configuring-the-shell.html
# On Windows, use PowerShell instead of sh.
set windows-shell := ["powershell.exe", "-c"]
# For the other platforms, use the default shell.

[doc('Run `just` without arguments to run all the main tasks.')]
default: update-env qa tests type-check build-docs

[doc('Run `just update-env` to update the virtual environment.
This will update the dependencies in `uv.lock`.')]
update-env:
    @echo "Updating the virtual environment and dependencies..."
    uv self update
    uv lock --upgrade
    uv sync

[doc('Run `just qa` to run all the quality assurance checks.
This includes updating and running pre-commit hooks,
which includes linting and formatting.')]
qa:
    @echo "Running quality assurance checks..."
    uv run pre-commit-update
    uv run pre-commit run --all-files

[doc('Run `just tests` to run all the tests.
This includes the unit tests in the `tests` directory,
as well as the doctests in the codebase.')]
tests:
    @echo "Running tests..."
    uv run pytest tests -vv
    uv run pytest src/pyresiflex --doctest-modules -vv --ignore src/pyresiflex/data

[doc('Run `just tests-cov` to run all the tests with coverage.
This includes generating coverage reports in various formats.')]
tests-cov:
    @echo "Running tests with coverage..."
    uv run coverage run -m pytest tests -vv
    uv run coverage report --show-missing --omit="*/tests/*"
    uv run coverage xml -o coverage/coverage.xml --omit="*/tests/*"
    uv run coverage html -d coverage/htmlcov --omit="*/tests/*"
    uv run genbadge coverage -i coverage/coverage.xml -o coverage/coverage.svg

[doc('Run `just type-check` to run the type checker.
This uses mypy to check the types in the codebase.
See https://mypy.readthedocs.io/en/stable/command_line.html for more information.')]
type-check:
    @echo "Running type checker..."
    uv run mypy . --exclude docs --exclude out --exclude data

[doc('Run `just build-docs` to build the documentation.
This uses Sphinx to build the docs in the `docs` directory.
See https://www.sphinx-doc.org/en/master/man/sphinx-build.html for more information.

You can pass the `rebuild` argument to specify which parts of the docs to rebuild.
By default, it will rebuild all the docs.
For example, to rebuild only the API docs, run `just build-docs _api`.
To rebuild all except examples, run `just build-docs "_api _build source/backreferences"`.')]
[working-directory: 'docs']
build-docs rebuild='all':
    @echo "Building documentation..."
    uv run python prepare_docs.py --folders {{rebuild}}
    uv run sphinx-build -M html . _build --fail-on-warning --nitpicky

[doc('Run `just stub-gen` to generate type hint stubs.')]
stub-gen:
    @echo "Generating type hint stubs..."
    uv run stubgen .\src\pyresiflex
