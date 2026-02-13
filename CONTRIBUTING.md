# Contributing to BarcodeBERT

Thank you for your interest in contributing to BarcodeBERT!
This document covers the standards and process for contributing.

## Getting started

### Prerequisites

- **Python 3.11 or 3.12** (`torchtext` lacks wheels for 3.13+)
- `pip` or [`uv`](https://docs.astral.sh/uv/)

### Local setup

Using `pip`, activate a virtual environment first, then:

```bash
git clone git@github.com:bioscan-ml/BarcodeBERT.git
cd BarcodeBERT
pip install -e .
```

Or, using `uv` (creates and manages the environment automatically):

```bash
git clone git@github.com:bioscan-ml/BarcodeBERT.git
cd BarcodeBERT
uv sync
```

Install the pre-commit hooks:

```bash
pip install pre-commit  # or: uv tool install pre-commit
pre-commit install
```

## Code style

Pre-commit hooks enforce all formatting automatically on commit.
You can also run them manually:

```bash
pre-commit run --all-files          # full repo
pre-commit run --files <file> ...   # specific files
```

Key settings:

| Tool   | Config           | Notes                             |
| ------ | ---------------- | --------------------------------- |
| black  | `pyproject.toml` | Line length 120                   |
| isort  | pre-commit       | `--profile=black`                 |
| flake8 | `.flake8`        | Line length 140, numpy docstrings |

### Flake8 suppressions

Certain warnings are suppressed project-wide (see `.flake8`):

- **E203**: whitespace before `:` (conflicts with black)
- **E402**: module-level import not at top (lazy imports)
- **E731**: lambda assignments
- **D100–D107**: missing docstrings

## Repository structure

The `barcodebert/` directory is the core Python package.
The editable install (`pip install -e .` or `uv sync`)
makes it importable so that scripts elsewhere in the repo
can use it.
The `baselines/` directory contains standalone evaluation scripts
that are **not** part of the package —
they import from `barcodebert` but are run directly.

## Contribution process

1. **Open an issue** for significant changes
   (bug reports, feature proposals, refactors).
2. **Create a feature branch** from `main`
   using a conventional prefix
   (e.g., `fix/`, `feat/`, `docs/`).
3. **Make your changes** following the code style above.
4. **Run pre-commit** to ensure formatting passes.
5. **Submit a pull request** against `main` with:
   - A clear description of the change
   - Reference to the related issue (if any)

### Commit messages

This project follows the
[NumPy-style commit message convention](https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message):

```
PREFIX: Short description
```

Common prefixes
(following [NumPy's list](https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message)):

| Prefix | Meaning                                 |
| ------ | --------------------------------------- |
| `API`  | An (incompatible) API change            |
| `BUG`  | Bug fix                                 |
| `CI`   | Continuous integration                  |
| `DEP`  | Deprecate something, or remove a deprecated object |
| `DEV`  | Development tool or utility             |
| `DOC`  | Documentation                           |
| `ENH`  | Enhancement                             |
| `MNT`  | Maintenance (refactoring, typos, etc.)  |
| `REL`  | Related to releasing                    |
| `REV`  | Revert an earlier commit                |
| `STY`  | Style fix (whitespace, PEP 8)           |
| `TST`  | Addition or modification of tests       |
| `TYP`  | Static typing                           |
| `WIP`  | Work in progress, do not merge          |

### Pull request guidelines

- Keep PRs focused on a single topic.
- Ensure pre-commit checks pass.
- PRs are squash-merged to maintain a clean `main` history.

## Dependency notes

Version constraints in `pyproject.toml` exist
for specific compatibility reasons:

- `datasets>=2.16,<4`:
  The HuggingFace dataset uses a custom loading script
  removed in `datasets` v4+.
- `numba>=0.59`:
  Prevents the `uv` resolver from backtracking
  to old `llvmlite` versions incompatible with Python 3.11+.
- `torchtext>=0.15.2`:
  Deprecated and archived; constrains Python to <3.13.

See [issue #21](https://github.com/bioscan-ml/BarcodeBERT/issues/21)
for background.
