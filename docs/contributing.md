# Developing torch-mentor

This page covers everything needed to work on the mentor codebase itself:
cloning, installing dev dependencies, running the test suite, building docs,
and publishing a release to PyPI.

## Clone and install

```bash
git clone https://github.com/anguelos/torch_mentor
cd torch_mentor
```

Create and activate a virtual environment, then install the package in editable
mode with all development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To also install documentation dependencies:

```bash
pip install -e ".[dev,docs]"
```

## Running the test suite

Run the full test suite (unit + integration + CLI tests):

```bash
make test
# equivalent to:
pytest tests/ -q
```

Run only unit tests with a coverage report:

```bash
make unittest
# equivalent to:
pytest tests/unit_testing/ -q --cov=mentor --cov-report=term-missing
```

Run a specific test file or test function:

```bash
pytest tests/unit_testing/test_mentee.py -v
pytest tests/unit_testing/test_mentee.py::test_create_train_objects -v
```

### Test layout

| Directory | What it covers |
|---|---|
| `tests/unit_testing/` | Individual classes and functions in isolation |
| `tests/checkpoint/` | Checkpoint save / resume round-trips |
| `tests/integration/` | End-to-end training loops |
| `tests/cli/` | `mtr_checkpoint` and `mtr_plot_file_hist` entry-points |

## Building the docs locally

```bash
make docs
# opens at docs/_build/html/index.html
```

Build a single-page HTML version:

```bash
make docs_single
```

Build a PDF (requires a LaTeX installation):

```bash
make docs_pdf
```

## Code style

The project uses [ruff](https://docs.astral.sh/ruff/) for linting, configured
in `pyproject.toml` with a 160-character line limit.  It is installed with
`.[dev]`.

Check for issues without touching any files:

```bash
make testlint
# equivalent to: ruff check mentor/ tests/
```

Auto-fix everything ruff can fix safely (unused imports, import order, etc.):

```bash
make autolint
# equivalent to: ruff check --fix mentor/ tests/
```

Type checking:

```bash
mypy mentor/
```

## Versioning

The version string lives in two places — keep them in sync before a release:

- `setup.py` -> `version=`
- `mentor/__init__.py` -> `__version__`

## Publishing to PyPI

Build source and wheel distributions:

```bash
pip install build twine
python -m build
```

Check the distributions before uploading:

```bash
twine check dist/*
```

Upload to PyPI (you need a PyPI account and an API token):

```bash
twine upload dist/*
```

Or upload to TestPyPI first to verify everything looks right:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ torch-mentor
```

The package is published at <https://pypi.org/project/torch-mentor/>.
