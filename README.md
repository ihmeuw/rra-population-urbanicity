# RRA Population Urbanicity

---

**Documentation**: [https://ihmeuw.github.io/rra-population-urbanicity](https://ihmeuw.github.io/rra-population-urbanicity)

**Source Code**: [https://github.com/ihmeuw/rra-population-urbanicity](https://github.com/ihmeuw/rra-population-urbanicity)

---

Classifies settlement type (urban/rural, Degree of Urbanisation) from gridded population and tabulates urban/rural population by administrative unit.

## Installation

The package is not published to PyPI. Install it from GitHub:

```sh
pip install git+https://github.com/ihmeuw/rra-population-urbanicity
```

To install a specific release, append its version, e.g. `@0.1.0`.

## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.10+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the `docs` directory and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Template and dependency updates

Neither runs on a schedule; both are done by hand, when wanted.

* **Template.** The repository was generated with [cruft](https://cruft.github.io/cruft/) from
  `collijk/python-package-cookiecutter`, at the commit recorded in `.cruft.json`. To pick up template
  changes, install cruft (`pip install cruft`) and, from a clean working tree, run `cruft check` to see
  whether the template has moved, then `cruft update` to apply it. Resolve any `.rej` files it leaves;
  the Test workflow fails while they exist.
* **Dependencies.** For a report of outdated direct dependencies, trigger the
  [Outdated dependencies workflow](https://github.com/ihmeuw/rra-population-urbanicity/actions/workflows/dependencies.yml)
  (press _Run workflow_); the table is on the run's summary page, and
  `poetry show --outdated --top-level` prints the same list locally. It includes versions that
  `pyproject.toml` blocks on purpose, such as the commented pre-commit cap. To update, run
  `poetry update <package>` within the current constraints, or edit a constraint and run `poetry lock`
  then `poetry install`; then run the tests.

### Releasing

Trigger the [Draft release workflow](https://github.com/ihmeuw/rra-population-urbanicity/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/ihmeuw/rra-population-urbanicity/releases) and publish it. When
 a release is published, it'll trigger the [release](https://github.com/ihmeuw/rra-population-urbanicity/blob/main/.github/workflows/release.yml) workflow, which
 deploys the updated documentation. Nothing is published to PyPI.

Both workflows use GitHub's built-in `GITHUB_TOKEN`, so the repository needs no token secrets. Two
consequences: the draft release pushes its version-bump commit straight to `main`, so it fails if `main`
is protected to require pull requests; and pushes made with that token don't start other workflows, so
the "Release x.y.z" commit gets no Test run (it changes only the version and the changelog).

### Pre-commit

Pre-commit hooks run all the auto-formatting (`ruff format`), linters (e.g. `ruff` and `mypy`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---
