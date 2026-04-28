---
applyTo: ".github/workflows/**"
---

# CI Workflow Changes

Use these rules for files under `.github/workflows/`. Apply the shared guidance
from `.github/copilot-instructions.md` first, then this workflow-specific
guidance.

## Workflow Overview

This repository has seven GitHub Actions workflows in `.github/workflows/`. Most
workflows use concurrency groups to cancel in-progress runs when superseded,
except `release.yml` (no concurrency group) and `api-docs.yml` (which uses a
`pages` concurrency group that does not cancel in-progress runs).

### `quick-checks.yml`

Runs pre-commit hooks on every pull request and on `workflow_dispatch`. The
hooks (configured in
`.pre-commit-config.yaml`) are: Black, pylint, isort (with special no-sections
config for sample files), yamlfix, shellcheck, GitHub workflow/action schema
validation (`check-github-workflows`, `check-github-actions`), and a local
`samples-generation` hook that regenerates reST samples when
`khiops/samples/samples.py` or `khiops/samples/samples_sklearn.py` change.

### `tests.yml`

The main test suite. Triggers on PRs that touch `khiops/**/*.py`,
`tests/**/*.py`, `tests/resources/**` (excluding `tests/resources/**/*.md`), or
the workflow file itself. Also supports `workflow_dispatch`.

Three job groups:

- **`run`** (Linux matrix): Runs across Python 3.10–3.14 in custom Docker
  containers (`ghcr.io/khiopsml/khiops-python/khiopspydev-ubuntu22.04`). Each
  Python version uses a dedicated Conda environment with native Khiops.
  Coverage is collected with `coverage` and reported as XML. Test results use
  JUnit XML via `unittest-xml-reporting`.
- **`check-khiops-integration-on-linux`**: Runs integration tests on multiple
  Linux containers (ubuntu22.04, rocky8, rocky9, debian13). Validates Khiops
  status, runs samples, tests major-version mismatch detection with a
  `py3_khiops10_conda` environment, and runs the integration test suite.
- **`check-khiops-integration-on-windows`**: Installs Khiops Desktop via NSIS
  installer on Windows 2022 with Python 3.12. Runs integration tests and
  samples outside a Python virtual environment, then installs khiops-python
  inside a venv and validates the installation status.

**Expensive tests** (remote file access with S3/GCS/Azure): Skipped by default
on feature branches. Enabled on `main`/`main-v10` branches or via the
`run-expensive-tests` workflow dispatch input. These require GCP Workload
Identity Federation, a local fake S3 server, and Azure storage credentials.

**Environment variables**: `KHIOPS_SAMPLES_DIR` points to a checkout of
`khiopsml/khiops-samples`. `KHIOPS_PROC_NUMBER=4` forces MPI multi-process
execution. MPI oversubscribe flags are set for Open MPI 4.x and 5+.

### `pip.yml`

Builds an **sdist** package (no wheel) and tests it in Docker containers
(ubuntu22.04, rocky9, debian13). Triggers on:

- Tag pushes (any tag) — automatically publishes to GitHub Releases
- PRs touching `pyproject.toml`, `LICENSE.md`, or the workflow file
- `workflow_dispatch` with optional `pypi-target` choice (`None`, `testpypi`,
  `pypi`)

Publishing to TestPyPI/PyPI uses OIDC Trusted Publishing and requires the
corresponding GitHub environment (`testpypi` or `pypi`). Only runs for the
`KhiopsML` org on tag pushes.

### `release.yml`

Manual workflow that merges `dev` into `main`, tags the merge commit with the
provided version, and resets `dev` to `main`. Only triggered via
`workflow_dispatch` with a `version` input.

### `api-docs.yml`

Builds Sphinx documentation inside a dev Docker container. Triggers on:

- Tag pushes — builds docs and uploads a zip archive to GitHub Releases
- PRs touching `doc/**/*.rst`, `doc/create-doc`, `doc/clean-doc`, `doc/*.py`,
  `khiops/**/*.py`, or the workflow file
- `workflow_dispatch` with optional tutorial and samples revision inputs

Uses the `khiopspydev-ubuntu22.04` Docker image and runs
`./create-doc -t -d -g <revision>`. Uses a `pages` concurrency group that does
**not** cancel in-progress runs (to avoid interrupting production deployments).

### `dev-docker.yml`

Builds development Docker images for multiple OS targets (ubuntu22.04, rocky8,
rocky9, debian13) with configurable Khiops revision, server revision, Python
versions (3.10–3.14), and remote file driver versions (GCS, S3, Azure).
Triggers on PRs touching `packaging/docker/khiopspydev/Dockerfile.*` or the
workflow file, and on `workflow_dispatch`. Images are pushed to
`ghcr.io/khiopsml/khiops-python/khiopspydev-*` only when manually requested via
`push: true`. The `set-latest` flag only works on the `main` or `main-v10`
branches.

### `test-conda-forge-package.yml`

Manual-only workflow that tests the released `khiops` Conda package on the
`conda-forge` channel across a broad matrix: Python 3.10–3.14 × multiple OS
environments (Ubuntu 20.04/22.04/24.04, Rocky 8/9, Windows 2022/2025, macOS
14/15/15-Intel). Tests both normal Conda environments and "Conda-based
environments" (where `CONDA_PREFIX` is unset to simulate non-Conda invocation).

## Editing Rules

- Workflow YAML files are validated by pre-commit hooks
  (`check-github-workflows`, `check-github-actions`) and formatted by `yamlfix`.
- The dev Docker images are the test environment for both `tests.yml` and
  `pip.yml`. If you need new system dependencies in CI, they go into the
  Dockerfiles under `packaging/docker/khiopspydev/`.
- Test dependencies are in `test-requirements.txt` (`coverage`, `wrapt`).
  Package dependencies are extracted from `pyproject.toml` at CI time via
  `scripts/extract_dependencies_from_pyproject_toml.py`.
