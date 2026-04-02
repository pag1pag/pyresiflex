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
default: update-env qa tests type-check build-docs all-done

[doc('Run `just update-env` to update the virtual environment.
This will update the dependencies in `uv.lock`.')]
update-env:
    @echo 'print("\x1b[1;36;40m" + "Updating the virtual environment and dependencies..." + "\x1b[0m")' | uv run -
    uv self update
    uv lock --upgrade
    uv sync
[doc('Run `just update-env-codespace` to update the virtual environment in a codespace.
This will update the dependencies in `uv.lock`.')]
update-env-codespace:
    @echo 'print("\x1b[1;36;40m" + "Updating the virtual environment and dependencies..." + "\x1b[0m")' | uv run -
    pip install --upgrade pip
    pip install --upgrade uv
    uv lock --upgrade
    uv sync

[doc('Run `just qa` to run all the quality assurance checks.
This includes updating and running pre-commit hooks,
which includes linting and formatting.')]
qa:
    @echo 'print("\x1b[1;36;40m" + "Running quality assurance checks..." + "\x1b[0m")' | uv run -
    uv run pre-commit-update
    uv run pre-commit run --all-files

[doc('Run `just tests` to run all the tests.
This includes the unit tests in the `tests` directory,
as well as the doctests in the codebase.')]
tests:
    @echo 'print("\x1b[1;36;40m" + "Running tests..." + "\x1b[0m")' | uv run -
    uv run pytest tests -vv
    uv run pytest src/pyresiflex --doctest-modules -vv --ignore src/pyresiflex/data

[doc('Run `just tests-cov` to run all the tests with coverage.
This includes generating coverage reports in various formats.')]
tests-cov:
    @echo 'print("\x1b[1;36;40m" + "Running tests with coverage..." + "\x1b[0m")' | uv run -
    uv run coverage run -m pytest tests -vv
    uv run coverage report --show-missing --omit="*/tests/*"
    uv run coverage xml -o coverage/coverage.xml --omit="*/tests/*"
    uv run coverage html -d coverage/htmlcov --omit="*/tests/*"
    uv run genbadge coverage -i coverage/coverage.xml -o coverage/coverage.svg

[doc('Run `just type-check` to run the type checker.
This uses `ty` to check the types in the codebase.
See https://docs.astral.sh/ty/ for more information.')]
type-check:
    @echo 'print("\x1b[1;36;40m" + "Running type checker..." + "\x1b[0m")' | uv run -
    uv run ty check

[doc('Run `just build-docs` to build the documentation.
This uses Sphinx to build the docs in the `docs` directory.
See https://www.sphinx-doc.org/en/master/man/sphinx-build.html for more information.

You can pass the `rebuild` argument to specify which parts of the docs to rebuild.
By default, it will rebuild all the docs.
For example, to rebuild only the API docs, run `just build-docs _api`.
To rebuild all except examples, run `just build-docs "_api _build source/backreferences"`.')]
[working-directory: 'docs']
build-docs rebuild='all':
    @echo 'print("\x1b[1;36;40m" + "Building documentation..." + "\x1b[0m")' | uv run -
    uv run python prepare_docs.py --folders {{rebuild}}
    uv run sphinx-build -M html . _build --fail-on-warning --nitpicky

[doc('Run `just stub-gen` to generate type hint stubs.')]
stub-gen:
    @echo 'print("\x1b[1;36;40m" + "Generating type hint stubs..." + "\x1b[0m")' | uv run -
    uv run stubgen .\src\pyresiflex

[doc('Print that all tasks are complete.')]
all-done:
    @echo 'print("\x1b[1;32;40m" + "All tasks complete!" + "\x1b[0m")' | uv run -
